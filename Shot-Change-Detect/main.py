import sys
import argparse
import detector
import benchmark

parser = argparse.ArgumentParser(description='Simple Shot Change Detection using Python+OpenCV')
parser.add_argument('path', metavar='file_path', type=str,
                    help='Path to a video file / directory of raw frames')
parser.add_argument('--method', type=str, default='HSV', required=False,
        help='Method: Edge/HSV/RGB (default: HSV)')
parser.add_argument('--threshold', type=float, default=0.2, required=False,
        help='Threshold of shot change (range: [0,1) )')
parser.add_argument('--min_length', type=int, default=12, required=False,
                    help='Minimum frame length of one shot')
parser.add_argument('--read_type', type=str, default='video', required=False,
        help='Read from raw frames (from directory) or a video file [video/dir] (default: video)')
parser.add_argument('--ground_truth', type=str, default='', required=False,
        help='Path to ground truth data')
parser.add_argument('--keyframe_out', type=str, default='', required=False,
        help='Path to generated keyframes')
parser.add_argument('--keyframe_threshold', type=float, default=0.25, required=False,
        help='Keyframe threshold (range: [0,1) )')
parser.add_argument('--scale', type=float, default=None, required=False,
        help='Subsampling')
parser.add_argument('--benchmark', action='store_true', default=False,
                    help='Do benchmark. If --method is set to \'all\' then run a full benchmark to compare between different methods.')
parser.add_argument('--benchmark_hard', action='store_true', default=False,
                    help='')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.method=='HSV':
        SCD = detector.ContentBased(directory = args.path, img_type=args.read_type, scale=args.scale)
    elif args.method=='RGB':
        SCD = detector.RGBBased(directory = args.path, img_type=args.read_type, scale=args.scale)
    else:
        SCD = detector.EdgeBased(directory = args.path, img_type=args.read_type, scale=args.scale)
    if not args.benchmark:
        sc = SCD.run(threshold = args.threshold, min_length = args.min_length)
        print('id, frame_n')
        for i,f in enumerate(sc): print('%d, %d'%(i,f))
        if len(args.keyframe_out)>0:
            SCD.get_keyframe(args.keyframe_threshold, args.keyframe_out)
    else:
        ground_truth = benchmark.gt_parser(args.ground_truth, args.benchmark_hard)
        if args.method=='all':
            methods = {
                    'Edge':SCD,
                    'HSV':detector.ContentBased(directory = args.path, img_type=args.read_type, scale=args.scale),
                    'RGB':detector.RGBBased(directory = args.path, img_type=args.read_type, scale=args.scale),
                    'HSVL2':detector.HSV2(directory = args.path, img_type=args.read_type, scale=args.scale),
                    'HSVL1':detector.HSV2(directory = args.path, img_type=args.read_type, scale=args.scale),
                    'RGBL1':detector.RGB1(directory = args.path, img_type=args.read_type, scale=args.scale),
                    'RGBL2':detector.RGB2(directory = args.path, img_type=args.read_type, scale=args.scale),
                    }
            benchmarker = benchmark.benchmark_plot_all(methods, ground_truth, args.min_length, SCD.get_frame_num())
            benchmarker.run()
        else:
            benchmarker = benchmark.benchmark(SCD, ground_truth, args.min_length)
            benchmarker.run(plot=True)
