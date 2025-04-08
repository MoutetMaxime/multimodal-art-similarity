import os
from typing import List
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


def download_image(image_url: str, save_folder: str, image_id: str, verbose: bool=True) -> None:
    """
    Download a single image from the given URL and save it to the specified folder.
    """
    image_name = f"{image_id}.jpg"

    # Check if the image already exists
    if os.path.exists(os.path.join(save_folder, image_name)):
        if verbose:
            print(f"Image {image_name} already exists. Skipping download.")
        return

    image_path = os.path.join(save_folder, image_name)

    # Get the image content and save it to the specified folder
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        with open(image_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        if verbose:
            print(f"Image downloaded and saved as {image_path}")
    except requests.RequestException as e:
        if verbose:
            print(f"Failed to download image for ID {image_id}: {e}")

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
        # If no single image is found, try to extract slider images
        slider_image_urls = extract_slider_image_urls(page_url)

        if slider_image_urls:
            # If slider images are found, it's from the "mnr" dataset
            if only_first:
                slider_image_dir = os.path.join(download_dir, f"mnr")
            else:
                slider_image_dir = os.path.join(download_dir, f"mnr/{image_id}")

            os.makedirs(slider_image_dir, exist_ok=True)
            for i, img_url in enumerate(slider_image_urls):
                if only_first and i == 0:
                    filename = str(image_id)
                    download_image(img_url, slider_image_dir, filename)
                    break

                filename = os.path.basename(img_url)
                download_image(img_url, slider_image_dir, filename.split('.')[0])

    if verbose:
        print(f"Finished. Images saved to {download_dir}/")


def download_lostart_csv(download_dir: str="../data/lostart", verbose: bool=False):
    """
    Download the lostart.csv file from the given URL and save it to the specified folder.
    """
    os.makedirs(download_dir, exist_ok=True)

    start = 0
    csv_url = "https://www.lostart.de/de/search-export/csv?start=0&filter%5Btype%5D%5B0%5D=Objektdaten&filter%5Bobject_type%5D%5Bpath%5D=Malerei"
    csv_path = os.path.join(download_dir, f"lostart_start={start}.csv")

    while True:
        try:
            response = requests.get(csv_url)
            response.raise_for_status()

            # Check if the response is only one line (indicating no more data)
            if len(response.text.strip()) < 200:
                if verbose:
                    print("No more data available.")
                break

            # Save the CSV file
            if os.path.exists(csv_path):
                break
            else:
                # Write the CSV content to a file
                with open(csv_path, 'wb') as f:
                    f.write(response.content)

            if verbose:
                print(f"CSV file downloaded and saved as {csv_path}")

        except requests.RequestException as e:
            if verbose:
                print(f"Failed to download CSV file: {e}")
            break

        # Increment the start parameter in the URL
        csv_url = csv_url.replace(f"start={start}", f"start={start + 500}")
        csv_path = os.path.join(download_dir, f"lostart_start={start + 500}.csv")

        start += 500


if __name__ == "__main__":
    # Example usage
    download_dir = "../../data/lostart"
    download_lostart_csv(download_dir, verbose=True)
