import os
from io import BytesIO
from typing import List
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from PIL import Image
from tqdm import tqdm


def download_image(image_url: str, save_folder: str, image_id: str, timeout: int = 60, verbose: bool = True) -> None:
    image_name = f"{image_id}.jpg"
    image_path = os.path.join(save_folder, image_name)

    if os.path.exists(image_path):
        if verbose:
            print(f"Image {image_name} already exists. Skipping.")
        return

    try:
        response = requests.get(image_url, stream=True, timeout=timeout)
        response.raise_for_status()

        with open(image_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)

        if verbose:
            print(f"Image downloaded and saved as {image_path}")

    except requests.exceptions.Timeout:
        raise  # Let the caller handle it
    except requests.RequestException as e:
        if verbose:
            print(f"Failed to download image {image_id}: {e}")


def download_image_in_memory(image_url: str):
    """
    Stream an image from the URL into memory and return a PIL Image.
    """
    response = requests.get(image_url, stream=True)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert('RGB')  # Always convert to RGB
    return img


def extract_single_image_url(page_url: str, verbose: bool=True) -> str:
    """
    Extract the single image URL from the given page URL.
    If not found, return an empty string.
    """
    try:
        response = requests.get(page_url)
        response.raise_for_status()
    except requests.RequestException as e:
        if verbose:
            print(f"Error retrieving page: {page_url} - {e}")
        return ""

    soup = BeautifulSoup(response.text, 'html.parser')

    # In our case, the image is inside a div with class "field field--name-field-media-image field--label-hidden"
    img_tag = soup.find("div", class_="field field--name-field-media-image field--label-hidden")

    # If the div is found, get the image tag inside it
    if img_tag:
        img_tag = img_tag.find("img")
    
    if img_tag and img_tag.get('src'):
        image_url = urljoin(page_url, img_tag['src'])
        return image_url
    else:
        if verbose:
            print(f"No single image found for the page: {page_url}")
        return ""


def extract_slider_image_urls(page_url: str, verbose: bool=True) -> List[str]:
    """
    Extract all image URLs from the slider on the given page URL.
    If no images are found, return an empty list.
    """

    # Get the page content
    try:
        response = requests.get(page_url)
        response.raise_for_status()
    except requests.RequestException as e:
        if verbose:
            print(f"Error retrieving page: {page_url} - {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')

    # In our case, the images are inside a div with class "swiper-slide"
    slider_images = soup.find_all("div", class_="swiper-slide")

    image_urls = []
    for slide in slider_images:
        img_tag = slide.find("img")
        if img_tag and img_tag.get('src'):
            img_url = urljoin(page_url, img_tag['src'])
            image_urls.append(img_url)

    if not image_urls and verbose:
        print(f"No slider images found for the page: {page_url}")

    return image_urls


def download_slider_from_page(page_url: str, image_id: str, download_dir: str = "data/images", only_first: bool = True, verbose: bool = False):
    os.makedirs(download_dir, exist_ok=True)

    # Extract image URLs from the page
    slider_image_urls = extract_slider_image_urls(page_url)

    if slider_image_urls:
        if only_first:
            slider_image_dir = os.path.join(download_dir, "mnr")
        else:
            slider_image_dir = os.path.join(download_dir, f"mnr/{image_id}")

        os.makedirs(slider_image_dir, exist_ok=True)

        for i, img_url in enumerate(slider_image_urls):
            try:
                if only_first and i == 0:
                    filename = str(image_id)
                    download_image(img_url, slider_image_dir, filename, timeout=60, verbose=verbose)
                    break

                filename = os.path.basename(img_url)
                download_image(img_url, slider_image_dir, filename.split('.')[0], timeout=60, verbose=verbose)

            except requests.exceptions.Timeout:
                if verbose:
                    print(f"⏱️ Timeout while downloading {img_url}, skipping to next.")
                continue
            except Exception as e:
                if verbose:
                    print(f"⚠️ Error while downloading {img_url}: {e}")
                continue

def download_images_from_page(page_url: str, image_id: str, download_dir: str="data/images", only_first: bool=True, verbose: bool=False):
    """
    Download images from a given page URL based on the image ID."
    """
    os.makedirs(download_dir, exist_ok=True)

    # Start by checking if it's a single image or a slider
    single_image_url = extract_single_image_url(page_url)

    if single_image_url:
        # If a single image is found, it's from the "lostart" dataset
        single_image_dir = os.path.join(download_dir, "lostart")
        os.makedirs(single_image_dir, exist_ok=True)
        download_image(single_image_url, single_image_dir, image_id)
    else:
        download_slider_from_page(page_url, image_id, download_dir, only_first, verbose)

    if verbose:
        print(f"Finished. Images saved to {download_dir}/")


def download_lostart_csv(download_dir: str = "../data/lostart", verbose: bool = False):
    """
    Download the lostart.csv files from the given URL and save them to the specified folder.
    Progress is displayed using tqdm.

    Parameters
    ----------
    download_dir : str
        Directory where the CSV files will be saved.
    verbose : bool
        Whether to print detailed download messages.
    """
    os.makedirs(download_dir, exist_ok=True)

    start = 0
    base_url = "https://www.lostart.de/de/search-export/csv?start={start}&filter%5Btype%5D%5B0%5D=Objektdaten"

    pbar = tqdm(desc="Downloading CSV files", unit="file")

    while True:
        csv_url = base_url.format(start=start)
        csv_path = os.path.join(download_dir, f"lostart_start={start}.csv")

        try:
            response = requests.get(csv_url)
            response.raise_for_status()

            if len(response.text.strip()) < 200:
                if verbose:
                    print("No more data available.")
                break

            if os.path.exists(csv_path):
                if verbose:
                    print(f"File already exists: {csv_path}")
                start += 500
                pbar.update(1)
                continue

            with open(csv_path, 'wb') as f:
                f.write(response.content)

            if verbose:
                print(f"CSV file downloaded and saved as {csv_path}")

        except requests.RequestException as e:
            if verbose:
                print(f"Failed to download CSV file: {e}")
            break

        start += 500
        pbar.update(1)

    pbar.close()



if __name__ == "__main__":
    download_dir = "data/lostart_2"
    download_lostart_csv(download_dir, verbose=False)
