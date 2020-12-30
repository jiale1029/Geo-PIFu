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
        for i in range(30):
            if args.config:
                cmd = (
                    "python render_mesh.py --meshDirSearch %sdeephuman_dataset --bgDirSearch " %(DATA_FOLDER) +
                    "%slsun --saveDir %shumanRender " %(DATA_FOLDER, DATA_FOLDER) +
                    "--resolutionScale 4 --useConfig --additionalType smplSemVoxels " +
                    "--splitNum 30 --splitIdx %s" %(i)
                )
            else:
                cmd = (
                    "python render_mesh.py --meshDirSearch %sdeephuman_dataset --bgDirSearch " %(DATA_FOLDER) +
                    "%slsun --saveDir %shumanRender_no_config " %(DATA_FOLDER, DATA_FOLDER) +
                    "--resolutionScale 4 --splitNum 30 --splitIdx %s" %(i)
                )
            cmds.append(cmd)
    elif args.prepare_query:
        for i in range(32):
            cmd = (
                "python -m apps.prepare_shape_query --sampleType occu_sigma3.5_pts5k " +
                "--shapeQueryDir /mnt/tanjiale/geopifu_dataset/shape_query " +
                "--datasetDir %shumanRender_no_config --epoch_range 0 15 " %(DATA_FOLDER) +
                "--sigma 3.5 --num_sample_inout 5000 --num_sample_color 0 --splitNum 32 --splitIdx %s" %(i)
            )
            cmds.append(cmd)
    elif args.prepare_voxel:
        DATASET_DIR = "/mnt/tanjiale/geopifu_dataset/humanRender_no_config"
        MESH_DIR = "/mnt/tanjiale/geopifu_dataset/deephuman_dataset"
        NAME = args.name

        for i in range(12):
            cmd = (
                "python -m apps.test_shape_coarse --datasetDir %s " %(DATASET_DIR)+
                "--resultsDir %s/geogeopifuResults/%s/train " %(DATASET_DIR,NAME)+
                "--splitNum 12 --splitIdx %s --gpu_id 1 " %(i)+
                "--load_netV_checkpoint_path ./checkpoints/%s/netV_epoch_29_2899 " %(NAME)+
                "--load_from_multi_GPU_shape --dataType train --batch_size 1 --mini_dataset"
            )
            cmds.append(cmd)

    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cpu_count)
    print(pool.map(work, cmds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pr", "--prepare_render", action="store_true", help="Render mesh and voxelization")
    parser.add_argument("-pq", "--prepare_query", action="store_true", help="Query Points Offline Sampling")
    parser.add_argument("-pv", "--prepare_voxel", action="store_true", help="Prepare aligned-latent-voxels for learning the final implicit function.")
    parser.add_argument("-c", "--config", action="store_true", help="With or without config")
    parser.add_argument("-n", "--name", default="GeoPIFU_course", help="Name of the network")

    args = parser.parse_args()

    assert( args.prepare_render or args.prepare_query or (args.prepare_voxel and args.name))
    main(args)
