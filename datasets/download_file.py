import logging
import os

import requests
from tqdm import tqdm

from aprec.utils.os_utils import mkdir_p_local, get_dir
def download_file(url, filename, data_dir):
    mkdir_p_local(data_dir)
    full_filename = os.path.join(get_dir(), data_dir, filename)
    if not os.path.isfile(full_filename):
        logging.info(f"downloading  {filename} file")
        response = requests.get(url, stream=True)
        with open(full_filename, 'wb') as out_file:
            expected_length = int(response.headers.get('content-length'))
            downloaded_bytes = 0
            with tqdm(total=expected_length, ascii=True) as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    out_file.write(chunk)
                    out_file.flush()
                    pbar.update(len(chunk))
        logging.info(f"{filename} dataset downloaded")
    else:
        logging.info(f"booking {filename} file already exists, skipping")
    return full_filename