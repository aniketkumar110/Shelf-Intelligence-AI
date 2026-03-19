"""  
Index ALL images from Azure Blob Storage into Azure AI Search using Azure AI Vision embeddings,  
then query by text to retrieve top related images (SAS URLs + optional download).  
  
Install:  
  pip install python-dotenv tenacity aiohttp azure-search-documents azure-storage-blob  
  
Required env:  
  AZURE_SEARCH_ENDPOINT1  
  AZURE_SEARCH_ADMIN_KEY1  
  AZURE_SEARCH_INDEX_NAME2  
  
  AZURE_AI_VISION_API_KEY  
  AZURE_AI_VISION_REGION  
  AZURE_AI_VISION_ENDPOINT   # optional  
  
Blob env (either):  
  AZURE_STORAGE_CONNECTION_STRING  
or:  
  AZURE_STORAGE_ACCOUNT_NAME  
  AZURE_STORAGE_ACCOUNT_KEY  
And:  
  AZURE_STORAGE_CONTAINER  
  
Optional:  
  EMBEDDING_CACHE_FILE=output/embedding_cache.pkl  
  MAX_CONCURRENT_REQUESTS=10  
  DOWNLOAD_WORKERS=12  
  LIST_BUFFER=50  
  UPLOAD_BATCH=200  
  VISION_THROTTLE_SLEEP=0.0  
"""  
  
import os  
import json  
import base64  
from unittest import result
import urllib.parse  
import argparse  
import logging  
import asyncio  
import aiohttp  
import pickle  
from pathlib import Path  
from datetime import datetime, timedelta  
from concurrent.futures import ThreadPoolExecutor  
from typing import Optional, Tuple, Dict, List  
  
from dotenv import load_dotenv  
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type  

from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient  
from azure.search.documents.indexes import SearchIndexClient  
from azure.search.documents.indexes.models import (  
    HnswParameters,  
    HnswAlgorithmConfiguration,  
    SimpleField,  
    SearchField,  
    SearchFieldDataType,  
    SearchIndex,  
    VectorSearch,  
    VectorSearchAlgorithmKind,  
    VectorSearchAlgorithmMetric,  
    VectorSearchProfile,  
)  
from azure.search.documents.models import VectorizedQuery  
  
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions  
  
  
# ---------------- logging ----------------  
logging.basicConfig(  
    level=logging.INFO,  
    format="%(asctime)s - %(levelname)s - %(message)s",  
    handlers=[logging.FileHandler("frames.log"), logging.StreamHandler()],  
)  
logger = logging.getLogger(__name__)  
  
load_dotenv()  
  
# ---------------- env ----------------  
SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT1", "").strip()  
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME2", "image_embeddings_index").strip()  
SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY1", "").strip()  
  
AI_VISION_KEY = os.getenv("AZURE_AI_VISION_API_KEY", "").strip()  
AI_VISION_REGION = os.getenv("AZURE_AI_VISION_REGION", "").strip()  
AI_VISION_ENDPOINT = os.getenv("AZURE_AI_VISION_ENDPOINT", "").strip()  
  
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "").strip()  
STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "").strip()  
STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY", "").strip()  
BLOB_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "").strip()  
  
CACHE_FILE = Path(os.getenv("EMBEDDING_CACHE_FILE", "output/embedding_cache.pkl"))  
EMBEDDING_CACHE: Dict[str, List[float]] = {}  
  
# Tuning  
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))  
DOWNLOAD_WORKERS = int(os.getenv("DOWNLOAD_WORKERS", "12"))  
LIST_BUFFER = int(os.getenv("LIST_BUFFER", "50"))     # blob names buffered before download/embed  
UPLOAD_BATCH = int(os.getenv("UPLOAD_BATCH", "200"))  # docs uploaded per batch  
VISION_THROTTLE_SLEEP = float(os.getenv("VISION_THROTTLE_SLEEP", "0.0"))  
  
  
# ---------------- helpers ----------------  
def _normalize_region(r: str) -> str:  
    return r.strip().lower().replace(" ", "") if r else r  
  
  
def safe_key(s: str) -> str:  
    return base64.urlsafe_b64encode(str(s).encode("utf-8")).decode("ascii")  
  
  
def _is_image_blob(name: str) -> bool:  
    n = (name or "").lower()  
    return n.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"))  
  
from azure.core.exceptions import ResourceNotFoundError  
  
def ensure_container_exists(bsc: BlobServiceClient, container: str):  
    cc = bsc.get_container_client(container)  
    try:  
        cc.get_container_properties()  
    except ResourceNotFoundError:  
        names = [c["name"] for c in bsc.list_containers()]  
        raise ValueError(  
            f"Container '{container}' not found in storage account '{bsc.account_name}'. "  
            f"Available containers: {names}"  
        )  
  
def _parse_conn_str(cs: str) -> dict:  
    parts = {}  
    for p in cs.split(";"):  
        if "=" in p:  
            k, v = p.split("=", 1)  
            parts[k.strip()] = v.strip()  
    return parts  
  
  
def _require_core_env():  
    missing = []  
    if not SERVICE_ENDPOINT:  
        missing.append("AZURE_SEARCH_ENDPOINT1")  
    if not SEARCH_ADMIN_KEY:  
        missing.append("AZURE_SEARCH_ADMIN_KEY1")  
    if not INDEX_NAME:  
        missing.append("AZURE_SEARCH_INDEX_NAME2")  
    if not AI_VISION_KEY:  
        missing.append("AZURE_AI_VISION_API_KEY")  
    if not AI_VISION_REGION:  
        missing.append("AZURE_AI_VISION_REGION")  
    if missing:  
        raise ValueError("Missing required env vars: " + ", ".join(missing))  
  
  
def _resolve_storage_creds_from_env():  
    """  
    Ensures:  
      - container exists in env  
      - we can create a BlobServiceClient  
      - if possible, we also derive account name/key for SAS generation  
    """  
    global STORAGE_ACCOUNT_NAME, STORAGE_ACCOUNT_KEY  
  
    if not BLOB_CONTAINER:  
        raise ValueError("Missing AZURE_STORAGE_CONTAINER")  
  
    if AZURE_STORAGE_CONNECTION_STRING:  
        parts = _parse_conn_str(AZURE_STORAGE_CONNECTION_STRING)  
        STORAGE_ACCOUNT_NAME = STORAGE_ACCOUNT_NAME or parts.get("AccountName", "")  
        STORAGE_ACCOUNT_KEY = STORAGE_ACCOUNT_KEY or parts.get("AccountKey", "")  
        return  
  
    if not (STORAGE_ACCOUNT_NAME and STORAGE_ACCOUNT_KEY):  
        raise ValueError(  
            "Provide AZURE_STORAGE_CONNECTION_STRING OR (AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY)"  
        )  
  
  
def get_blob_service_client() -> BlobServiceClient:  
    _resolve_storage_creds_from_env()  
    if AZURE_STORAGE_CONNECTION_STRING:  
        return BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)  
    return BlobServiceClient(  
        account_url=f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net",  
        credential=STORAGE_ACCOUNT_KEY,  
    )  
  
  
def blob_sas_url(blob_name: str, minutes: int = 30) -> Optional[str]:  
    """  
    Returns a read-only SAS URL if account name+key are available; otherwise None.  
    """  
    _resolve_storage_creds_from_env()  
    if not (STORAGE_ACCOUNT_NAME and STORAGE_ACCOUNT_KEY):  
        return None  
  
    sas = generate_blob_sas(  
        account_name=STORAGE_ACCOUNT_NAME,  
        account_key=STORAGE_ACCOUNT_KEY,  
        container_name=BLOB_CONTAINER,  
        blob_name=blob_name,  
        permission=BlobSasPermissions(read=True),  
        expiry=datetime.utcnow() + timedelta(minutes=minutes),  
    )  
    return (  
        f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net/"  
        f"{BLOB_CONTAINER}/{urllib.parse.quote(blob_name)}?{sas}"  
    )  
  
  
def load_embedding_cache():  
    if CACHE_FILE.exists():  
        try:  
            with open(CACHE_FILE, "rb") as f:  
                EMBEDDING_CACHE.update(pickle.load(f))  
            logger.info(f"Loaded embedding cache from {CACHE_FILE}")  
        except Exception as e:  
            logger.warning(f"Failed to load embedding cache: {e}")  
  
  
def save_embedding_cache():  
    try:  
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)  
        with open(CACHE_FILE, "wb") as f:  
            pickle.dump(EMBEDDING_CACHE, f)  
        logger.info(f"Saved embedding cache to {CACHE_FILE}")  
    except Exception as e:  
        logger.warning(f"Failed to save embedding cache: {e}")  
  
  
def _vision_url(region: str, endpoint: str, url_path: str) -> str:  
    if endpoint and endpoint.strip():  
        parsed = urllib.parse.urlparse(endpoint.strip())  
        if not parsed.netloc:  
            raise ValueError(f"Invalid AZURE_AI_VISION_ENDPOINT: {endpoint}")  
        return f"{endpoint.rstrip('/')}{url_path}"  
    host = f"{region}.api.cognitive.microsoft.com"  
    return f"https://{host}{url_path}"  
  
  
# ---------------- Vision embeddings ----------------  
@retry(  
    stop=stop_after_attempt(3),  
    wait=wait_fixed(1),  
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),  
)  
async def get_image_vector(  
    image_id: str,  
    image_bytes: bytes,  
    key: str,  
    region: str,  
    endpoint: Optional[str],  
    session: aiohttp.ClientSession,  
) -> List[float]:  
    cache_key = f"image::{image_id}::{region}"  
    if cache_key in EMBEDDING_CACHE:  
        return EMBEDDING_CACHE[cache_key]  
  
    headers = {"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/octet-stream"}  
    params = urllib.parse.urlencode({"model-version": "2023-04-15"})  
    url_path = f"/computervision/retrieval:vectorizeImage?api-version=2024-02-01&{params}"  
    url = _vision_url(region, endpoint or "", url_path)  
  
    async with session.post(url, headers=headers, data=image_bytes, timeout=60) as resp:  
        if resp.status != 200:  
            msg = await resp.text()  
            raise Exception(f"Vision API {resp.status}: {msg}")  
        data = await resp.json()  
  
    vec = data.get("vector")  
    if not isinstance(vec, list):  
        raise Exception(f"No 'vector' in response for {image_id}")  
  
    EMBEDDING_CACHE[cache_key] = vec  
    return vec  
  
  
@retry(  
    stop=stop_after_attempt(3),  
    wait=wait_fixed(1),  
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),  
)  
async def get_text_vector(  
    prompt: str,  
    key: str,  
    region: str,  
    endpoint: Optional[str],  
    session: aiohttp.ClientSession,  
) -> List[float]:  
    cache_key = f"text::{prompt}::{region}"  
    if cache_key in EMBEDDING_CACHE:  
        return EMBEDDING_CACHE[cache_key]  
  
    headers = {"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/json"}  
    params = urllib.parse.urlencode({"model-version": "2023-04-15"})  
    url_path = f"/computervision/retrieval:vectorizeText?api-version=2024-02-01&{params}"  
    url = _vision_url(region, endpoint or "", url_path)  
  
    async with session.post(url, headers=headers, json={"text": prompt}, timeout=60) as resp:  
        if resp.status != 200:  
            msg = await resp.text()  
            raise Exception(f"Vision API {resp.status}: {msg}")  
        data = await resp.json()  
  
    vec = data.get("vector")  
    if not isinstance(vec, list):  
        raise Exception("No 'vector' in response for text embedding")  
  
    EMBEDDING_CACHE[cache_key] = vec  
    return vec  
  
  
# ---------------- Azure AI Search index ----------------  
def reset_index(index_name: str = INDEX_NAME):  
    cred = AzureKeyCredential(SEARCH_ADMIN_KEY)  
    ic = SearchIndexClient(endpoint=SERVICE_ENDPOINT, credential=cred)  
    try:  
        ic.delete_index(index_name)  
        logger.info(f"Deleted index '{index_name}'")  
    except Exception:  
        pass  
  
  
def create_or_update_index(index_name: str = INDEX_NAME):  
    """  
    Index schema WITHOUT video_id.  
    If you previously had video_id in this same index name, use --reset-index.  
    """  
    cred = AzureKeyCredential(SEARCH_ADMIN_KEY)  
    ic = SearchIndexClient(endpoint=SERVICE_ENDPOINT, credential=cred)  
  
    fields = [  
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),  
        SearchField(name="container", type=SearchFieldDataType.String, filterable=True, sortable=True),  
        SearchField(name="blob_name", type=SearchFieldDataType.String, filterable=True, sortable=True),  
        SearchField(  
            name="image_vector",  
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),  
            searchable=True,  
            vector_search_dimensions=1024,  
            vector_search_profile_name="myHnswProfile",  
        ),  
    ]  
  
    vector_search = VectorSearch(  
        algorithms=[  
            HnswAlgorithmConfiguration(  
                name="myHnsw",  
                kind=VectorSearchAlgorithmKind.HNSW,  
                parameters=HnswParameters(  
                    m=4,  
                    ef_construction=400,  
                    ef_search=1000,  
                    metric=VectorSearchAlgorithmMetric.COSINE,  
                ),  
            )  
        ],  
        profiles=[VectorSearchProfile(name="myHnswProfile", algorithm_configuration_name="myHnsw")],  
    )  
  
    idx = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)  
    ic.create_or_update_index(idx)  
    logger.info(f"Index '{index_name}' created/updated.")  
  
  
# ---------------- Index ALL images from Blob ----------------  
async def index_all_images_from_blob(prefix: str = "", container: str = BLOB_CONTAINER):  
    _require_core_env()  
    _resolve_storage_creds_from_env()  
    if not container:  
        raise ValueError("Container is empty. Set AZURE_STORAGE_CONTAINER or pass --container")  
  
    load_embedding_cache()  
  
    region = _normalize_region(AI_VISION_REGION)  
    bsc = get_blob_service_client()  
    ensure_container_exists(bsc, container)  
    cc = bsc.get_container_client(container)  
  
    search_cred = AzureKeyCredential(SEARCH_ADMIN_KEY)  
    search_client = SearchClient(endpoint=SERVICE_ENDPOINT, index_name=INDEX_NAME, credential=search_cred)  
  
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)  
  
    def download_blob_bytes(blob_name: str) -> Tuple[str, Optional[bytes]]:  
        # If already embedded, skip download  
        cache_key = f"image::{blob_name}::{region}"  
        if cache_key in EMBEDDING_CACHE:  
            return blob_name, b""  # sentinel means "already cached"  
        try:  
            bc = bsc.get_blob_client(container=container, blob=blob_name)  
            return blob_name, bc.download_blob().readall()  
        except Exception as e:  
            logger.error(f"Download failed for blob '{blob_name}': {e}")  
            return blob_name, None  
  
    async with aiohttp.ClientSession() as session:  
        docs_batch = []  
  
        async def embed_one(blob_name: str, content: bytes) -> dict:  
            async with sem:  
                if content == b"":  
                    vec = EMBEDDING_CACHE[f"image::{blob_name}::{region}"]  
                else:  
                    vec = await get_image_vector(  
                        blob_name, content, AI_VISION_KEY, region, AI_VISION_ENDPOINT, session  
                    )  
  
                if VISION_THROTTLE_SLEEP > 0:  
                    await asyncio.sleep(VISION_THROTTLE_SLEEP)  
  
                return {  
                    "id": safe_key(f"{container}::{blob_name}"),  
                    "container": container,  
                    "blob_name": blob_name,  
                    "image_vector": vec,  
                }  
  
        def flush_upload():  
            nonlocal docs_batch  
            if not docs_batch:  
                return  
            try:  
                search_client.upload_documents(documents=docs_batch)  
                logger.info(f"Uploaded {len(docs_batch)} docs")  
            except Exception as e:  
                logger.error(f"Upload batch failed: {e}")  
            finally:  
                docs_batch = []  
  
        blob_buf: List[str] = []  
        total_seen = 0  
        total_indexed = 0  
  
        for blob in cc.list_blobs(name_starts_with=prefix):  
            total_seen += 1  
            if not _is_image_blob(blob.name):  
                continue  
  
            blob_buf.append(blob.name)  
  
            if len(blob_buf) >= LIST_BUFFER:  
                with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:  
                    downloaded = list(ex.map(download_blob_bytes, blob_buf))  
                blob_buf.clear()  
  
                tasks = [embed_one(bn, content) for bn, content in downloaded if content is not None]  
                results = await asyncio.gather(*tasks, return_exceptions=True)  
  
                for r in results:  
                    if isinstance(r, dict):  
                        docs_batch.append(r)  
                        total_indexed += 1  
                    else:  
                        logger.error(f"Embedding failed: {r}")  
  
                if len(docs_batch) >= UPLOAD_BATCH:  
                    flush_upload()  
  
        # leftovers  
        if blob_buf:  
            with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:  
                downloaded = list(ex.map(download_blob_bytes, blob_buf))  
            blob_buf.clear()  
  
            tasks = [embed_one(bn, content) for bn, content in downloaded if content is not None]  
            results = await asyncio.gather(*tasks, return_exceptions=True)  
  
            for r in results:  
                if isinstance(r, dict):  
                    docs_batch.append(r)  
                    total_indexed += 1  
                else:  
                    logger.error(f"Embedding failed: {r}")  
  
        flush_upload()  
        save_embedding_cache()  
        logger.info(f"Done. Seen={total_seen}, Indexed={total_indexed}, Prefix='{prefix}', Container='{container}'")  
  
  
def build_from_blob(prefix: str = "", container: str = BLOB_CONTAINER, reset: bool = False):  
    _require_core_env()  
    if reset:  
        reset_index(INDEX_NAME)  
    create_or_update_index(INDEX_NAME)  
    asyncio.run(index_all_images_from_blob(prefix=prefix, container=container))  

def index_single_blob_sync(blob_name: str, container: str = BLOB_CONTAINER):
    """
    Index only ONE blob into Azure AI Search.
    Used by Streamlit upload flow.
    """
    _require_core_env()
    _resolve_storage_creds_from_env()
    load_embedding_cache()

    create_or_update_index(INDEX_NAME)

    region = _normalize_region(AI_VISION_REGION)

    bsc = get_blob_service_client()
    bc = bsc.get_blob_client(container=container, blob=blob_name)
    image_bytes = bc.download_blob().readall()

    async def _embed_and_upload():
        async with aiohttp.ClientSession() as session:
            vec = await get_image_vector(
                blob_name,
                image_bytes,
                AI_VISION_KEY,
                region,
                AI_VISION_ENDPOINT,
                session,
            )

        search_client = SearchClient(
            endpoint=SERVICE_ENDPOINT,
            index_name=INDEX_NAME,
            credential=AzureKeyCredential(SEARCH_ADMIN_KEY),
        )

        doc = {
            "id": safe_key(f"{container}::{blob_name}"),
            "container": container,
            "blob_name": blob_name,
            "image_vector": vec,
        }

        result = search_client.upload_documents(documents=[doc])
        print("Upload result:", result)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_embed_and_upload())
    loop.close()
  
  
# ---------------- Query + retrieve top images from Blob ----------------  
def download_blob_to_file(container: str, blob_name: str, dest: Path) -> Path:  
    bsc = get_blob_service_client()  
    dest.parent.mkdir(parents=True, exist_ok=True)  
    bc = bsc.get_blob_client(container=container, blob=blob_name)  
    with open(dest, "wb") as f:  
        f.write(bc.download_blob().readall())  
    return dest  
  
  
def query_and_fetch_images(  
    prompt: str,  
    top_k: int = 5,  
    container: str = BLOB_CONTAINER,  
    fetch_dir: Path = Path("retrieved_images"),  
    download: bool = True,  
    sas_minutes: int = 30,  
):  
    _require_core_env()  
    _resolve_storage_creds_from_env()  
  
    load_embedding_cache()  
    region = _normalize_region(AI_VISION_REGION)  
  
    async def _embed_text():  
        async with aiohttp.ClientSession() as session:  
            return await get_text_vector(prompt, AI_VISION_KEY, region, AI_VISION_ENDPOINT, session)  
  
    query_vec = asyncio.run(_embed_text())  
  
    cred = AzureKeyCredential(SEARCH_ADMIN_KEY)  
    sc = SearchClient(endpoint=SERVICE_ENDPOINT, index_name=INDEX_NAME, credential=cred)  
  
    vq = VectorizedQuery(vector=query_vec, k_nearest_neighbors=top_k, fields="image_vector")  
    results = sc.search(  
        search_text="",  
        vector_queries=[vq],  
        select=["id", "container", "blob_name"],  
        top=top_k,  
    )  
  
    fetch_dir.mkdir(parents=True, exist_ok=True)  
    out = []  
  
    for r in results:  
        blob_name = r.get("blob_name")  
        cont = r.get("container") or container  
  
        sas_url = blob_sas_url(blob_name, minutes=sas_minutes) if blob_name else None  
  
        local_path = None  
        if download and blob_name:  
            dest = fetch_dir / blob_name  # preserves folder structure if blob_name has '/'  
            try:  
                download_blob_to_file(cont, blob_name, dest)  
                local_path = str(dest.resolve())  
            except Exception as e:  
                logger.error(f"Failed to download '{blob_name}': {e}")  
  
        out.append(  
            {  
                "score": r.get("@search.score"),  
                "container": cont,  
                "blob_name": blob_name,  
                "sas_url": sas_url,  
                "downloaded_file": local_path,  
            }  
        )  
  
    return out  
  
  
# ---------------- CLI ----------------  
def main():  
    parser = argparse.ArgumentParser(  
        description="Index images from Azure Blob -> vector search -> fetch top matching images from Blob"  
    )  
  
    parser.add_argument("--build-blob", action="store_true", help="Index ALL images from Blob Storage into Azure AI Search")  
    parser.add_argument("--reset-index", action="store_true", help="DELETE + recreate the index (needed if schema changed)")  
    parser.add_argument("--container", type=str, default=BLOB_CONTAINER, help="Blob container name")  
    parser.add_argument("--prefix", type=str, default="", help="Index only blobs under this prefix (e.g. 'cats/')")  
  
    parser.add_argument("--query", type=str, default=None, help="Text query to search similar images")  
    parser.add_argument("--top-k", type=int, default=5, help="How many results to return")  
    parser.add_argument("--fetch-dir", type=str, default="retrieved_images", help="Where to download top images")  
    parser.add_argument("--no-download", action="store_true", help="Don't download; only return SAS URLs (if possible)")  
    parser.add_argument("--sas-minutes", type=int, default=30, help="SAS URL expiry in minutes")  
  
    args = parser.parse_args()  
  
    if args.build_blob:  
        build_from_blob(prefix=args.prefix, container=args.container, reset=args.reset_index)  
  
    if args.query:  
        hits = query_and_fetch_images(  
            prompt=args.query,  
            top_k=args.top_k,  
            container=args.container,  
            fetch_dir=Path(args.fetch_dir),  
            download=(not args.no_download),  
            sas_minutes=args.sas_minutes,  
        )  
        print(json.dumps({"prompt": args.query, "top_k": args.top_k, "results": hits}, indent=2))  
  
    if not args.build_blob and not args.query:  
        parser.print_help()  
  
  
if __name__ == "__main__":  
    main()  

# python sidebarSearch.py --build-blob --reset-index --container vision-agent --prefix ""  
# python sidebarSearch.py --build-blob --reset-index --container images --prefix "cats/"  
# python sidebarSearch.py --query "a red car on a highway" --top-k 5 --fetch-dir retrieved_images  
# python sidebarSearch.py --query "a red car on a highway" --top-k 5 --no-download  

# python sidebarSearch.py --query "cold drinks" --top-k 5 --container vision-agent --fetch-dir retrieved_images  

# python sidebarSearch.py --query "side view of a shelf with products" --top-k 5 --container vision-agent --fetch-dir retrieved_images  
