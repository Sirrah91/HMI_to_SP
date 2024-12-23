from modules.align_data import pipeline_sp_to_hmilike, pipeline_alignment, pipeline_sp_to_hmilike_v2

import argparse
import socket


if __name__ == "__main__":
    hostname = socket.gethostname()
    print(f"Running on: {hostname}\n")

    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("--exact_hmi", action="store_true")
    parser.add_argument("--coordinates", type=str, default="ptr_native")
    parser.add_argument("--interpolate_outliers", action="store_true")
    parser.add_argument("--subpixel_shift", action="store_true")
    parser.add_argument("--cut_edge_px", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=72)
    parser.add_argument("--output_name", type=str, default="")

    args, _ = parser.parse_known_args()

    if args.exact_hmi:
        if not args.output_name:
             args.output_name = "SP_HMI_aligned.npz"
        pipeline_alignment(coordinates=args.coordinates,
                           interpolate_outliers=args.interpolate_outliers,
                           subpixel_shift=args.subpixel_shift,
                           patch_size=args.patch_size,
                           cut_edge_px=args.cut_edge_px,
                           output_name=args.output_name,
                           skip_jsoc_query=False,
                           alignment_plots=False,
                           control_plots=False,
                           do_parallel=False)
    else:
        if not args.output_name:
            args.output_name = "SP_HMI-like.npz"
        pipeline_sp_to_hmilike_v2(coordinates=args.coordinates,
                                  interpolate_outliers=args.interpolate_outliers,
                                  patch_size=args.patch_size,
                                  cut_edge_px=args.cut_edge_px,
                                  output_name=args.output_name,
                                  alignment_plots=False,
                                  control_plots=False,
                                  do_parallel=False)
