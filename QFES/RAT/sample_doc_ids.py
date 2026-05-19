import json
import random
from typing import List, Union


def sample_document_ids(
    json_file_path: str,
    sample_ratio: float = 0.10,
    seed: int = 42,
    id_field: str = "id",
    min_sample_size: int = 1
) -> List[Union[int, str]]:
    """
    Randomly sample document IDs from a JSON corpus.

    Args:
        json_file_path: Path to the input corpus JSON file.
        sample_ratio: Sampling ratio. Default is 0.10, meaning 10%.
        seed: Random seed for reproducibility.
        id_field: Field name used as the document ID. Default is "id".
        min_sample_size: Minimum number of sampled documents.

    Returns:
        A list of sampled document IDs.
    """
    if not 0 < sample_ratio <= 1:
        raise ValueError("sample_ratio must be in the range (0, 1].")

    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_ids = []

    for item in data:
        if id_field not in item:
            continue

        raw_id = item[id_field]

        try:
            raw_id = int(str(raw_id).split("#")[0])
        except Exception:
            raw_id = str(raw_id)

        all_ids.append(raw_id)

    if not all_ids:
        raise ValueError(f"No valid document IDs found using id_field='{id_field}'.")

    sample_size = max(min_sample_size, int(len(all_ids) * sample_ratio))
    sample_size = min(sample_size, len(all_ids))

    rng = random.Random(seed)
    sampled_ids = rng.sample(all_ids, sample_size)

    print(f"[INFO] Total documents: {len(all_ids)}")
    print(f"[INFO] Sampling ratio: {sample_ratio}")
    print(f"[INFO] Sampled documents: {len(sampled_ids)}")

    return sampled_ids