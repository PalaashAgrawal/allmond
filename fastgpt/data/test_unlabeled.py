from unlabeled import download_dataset
import datasets
from datasets import load_dataset


#write test case for download_dataset
def test_download_dataset():
    # Test the download_dataset function
    dataset = download_dataset("openai/gpt-2")
    assert isinstance(dataset, datasets.Dataset), f"Expected a datasets.Dataset object, but got {type(dataset)}"

    # Test the download_dataset function with an invalid dataset name
    try:
        download_dataset("invalid/dataset")
    except AssertionError as e:
        assert str(e) == "Dataset invalid/dataset not found", f"Expected an AssertionError, but got {e}"

    # Test the download_dataset function with multiple datasets
    dataset1, dataset2 = download_dataset("openai/gpt-2", "openai/gpt-3")
    assert isinstance(dataset1, datasets.Dataset), f"Expected a datasets.Dataset object, but got {type(dataset1)}"
    assert isinstance(dataset2, datasets.Dataset), f"Expected a datasets.Dataset object, but got {type(dataset2)}"
    
    

if __name__ == "__main__":
    test_download_dataset()
    print("download_dataset passed")    