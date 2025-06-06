"""Script to create data cache"""
import os

from trajdata import UnifiedDataset


def main():
    """Create data cache"""
    dataset = UnifiedDataset(
        desired_data=[
            "eupeds_eth",
            "eupeds_hotel",
            "eupeds_univ",
            "eupeds_zara1",
            "eupeds_zara2"
        ],
        rebuild_cache=True,
        rebuild_maps=True,
        num_workers=os.cpu_count(),
        verbose=True,
        data_dirs={  # Remember to change this to match your filesystem!
            "eupeds_eth": "./pedestrian_datasets/eth_ucy_peds",
            "eupeds_hotel": "./pedestrian_datasets/eth_ucy_peds",
            "eupeds_univ": "./pedestrian_datasets/eth_ucy_peds",
            "eupeds_zara1": "./pedestrian_datasets/eth_ucy_peds",
            "eupeds_zara2": "./pedestrian_datasets/eth_ucy_peds",
        },
    )
    print(f"Total Data Samples: {len(dataset):,}")


if __name__ == "__main__":
    main()
