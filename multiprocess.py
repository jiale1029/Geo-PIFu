import argparse
import multiprocessing
import subprocess

DATA_FOLDER = "/mnt/tanjiale/geopifu_dataset/"
CONFIG = False

def work(cmd):
    return subprocess.call(cmd, shell=True)

def main(args):
    cmds = []
    if args.prepare_render:
        split_num = 24 # specify your cpu count
        for i in range(split_num):
            if args.config:
                cmd = (
                    "python render_mesh.py " + 
                    "--meshDirSearch % " %(args.meshDirSearch) +
                    "--bgDirSearch %slsun " %(DATA_FOLDER) +
                    "--saveDir %s " %(args.datasetDir) +
                    "--resolutionScale 4 " +
                    "--useConfig " +
                    "--additionalType smplSemVoxels " +
                    "--splitNum %s " %(split_num) +
                    "--splitIdx %s" %(i)
                )
            else:
                cmd = (
                    "python render_mesh.py " +
                    "--meshDirSearch % " %(args.meshDirSearch) +
                    "--bgDirSearch %slsun " %(DATA_FOLDER) +
                    "--saveDir % " %(args.datasetDir) +
                    "--resolutionScale 4 " +
                    "--splitNum %s " %(split_num) +
                    "--splitIdx %s" %(i)
                )
            cmds.append(cmd)
    elif args.prepare_color:
        split_num = 12
        for i in range(split_num):
            cmd = (
                "python -m apps.prepare_color_query " +
                "--sampleType sigma0.005_pts8k " +
                "--shapeQueryDir %s " %(args.shapeQueryDir) +
                "--datasetDir %s " %(args.datasetDir) +
                "--epoch_range 0 15 " +
                "--sigma 0.005 " +
                "--num_sample_color 8000 " +
                "--splitNum %s " %(split_num) +
                "--splitIdx %s " %(i)
            )
            cmds.append(cmd)
    elif args.prepare_query:
        split_num = 24
        for i in range(split_num):
            cmd = (
                "python -m apps.prepare_shape_query " +
                "--sampleType occu_sigma3.5_pts5k " +
                "--shapeQueryDir %s " %(args.shapeQueryDir) +
                "--datasetDir %s " %(args.datasetDir) +
                "--epoch_range 0 15 " +
                "--sigma 3.5 " +
                "--num_sample_inout 5000 " +
                "--num_sample_color 0 " +
                "--splitNum %s " %(split_num)+
                "--splitIdx %s" %(i)
            )
            cmds.append(cmd)
    elif args.prepare_voxel:
        RESULT_DIR = args.resultsDir
        DATASET_DIR = args.datasetDir
        MESH_DIR = args.meshDirSearch
        NAME = args.name

        for i in range(2):
            cmd = (
                "python -m apps.test_shape_coarse " +
                "--datasetDir %s " %(DATASET_DIR) +
                "--resultsDir %s/%s/train " %(RESULT_DIR,NAME) +
                "--splitNum %s " %(str(2)) +
                "--splitIdx %s " %(str(i)) +
                "--gpu_id %s " %(str(args.gpu_id)) +
                "--load_netV_checkpoint_path %s " %(args.netV_checkpoint)+
                "--load_from_multi_GPU_shape " +
                "--dataType %s " %(args.dataType) +
                "--batch_size 1 " +
                "--datasetType %s" %(args.datasetType)
            )
            print(cmd)
            cmds.append(cmd)

    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cpu_count)
    print(pool.map(work, cmds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pr", "--prepare_render", action="store_true", help="Render mesh and voxelization")
    parser.add_argument("-pq", "--prepare_query", action="store_true", help="Query Points Offline Sampling")
    parser.add_argument("-pc", "--prepare_color", action="store_true", help="Color Points Offline Sampling")
    parser.add_argument("-pv", "--prepare_voxel", action="store_true", help="Prepare aligned-latent-voxels for learning the final implicit function.")
    parser.add_argument("-c", "--config", action="store_true", help="With or without config")
    parser.add_argument("-n", "--name", default="GeoPIFu_coarse", help="Name of the network")
    parser.add_argument("--shapeQueryDir", default="/mnt/tanjiale/geopifu_dataset/shape_query", help="Sampled point directory")
    parser.add_argument("--resultsDir", default="/mnt/tanjiale/geopifu_dataset/geopifu_results", help="Directory of the result")
    parser.add_argument("--datasetDir", default="/mnt/tanjiale/geopifu_dataset/humanRender_no_config", help="Directory of the processsed deephuman dataset")
    parser.add_argument("--meshDirSearch", default="/mnt/tanjiale/geopifu_dataset/deephuman_dataset", help="Directory of the deephuman dataset")
    parser.add_argument("--dataType", default="train", help="Data for training or testing")
    parser.add_argument('--datasetType', type=str, default='all', help="all, mini, adjusted")
    parser.add_argument("--gpu_id", default=0, type=int, help="gpu_id")
    parser.add_argument("--netV_checkpoint", default="./checkpoints/geopifu_coarse/netV_epoch_29_191", help="Checkpoint for netV")

    args = parser.parse_args()

    assert( args.prepare_render or args.prepare_query or (args.prepare_voxel and args.name) or args.prepare_color)
    main(args)
