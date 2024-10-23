import os

from torchinfo import summary

from segone.networks.segone_network import SegOne

if __name__ == "__main__":
    """Test dummy inputs for structural testing"""
    print("Verify SegOne Classification Architecture")
    model = SegOne(
        {
            "type": "classification",
            "channel_in": 3,
            "channel_out": 13,
            "num_layers": 4,
            "bottleneck_scale": 2,
            "bottleneck_repeat": 3,
            "bottleneck_channel": 32,
            "kernel_size": 3,
        }
    ).cuda()
    summary(model, input_size=(1, 3, 512, 512))

    print("\n\n\n" + "".join(["#"] * 95) + "\n\n\n")

    print("Verify SegOne Segmentation Architecture")
    model = SegOne(
        {
            "type": "segmentation",
            "channel_in": 3,
            "channel_out": 13,
            "num_layers": 4,
            "bottleneck_scale": 2,
            "bottleneck_repeat": 3,
            "bottleneck_channel": 32,
            "kernel_size": 3,
        }
    ).cuda()
    summary(model, input_size=(1, 3, 512, 512))
