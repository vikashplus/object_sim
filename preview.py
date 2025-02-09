import mujoco
import time

from mujoco import viewer
import os
import click
import skvideo.io
from sys import platform
import numpy as np

def get_aed(lookat: np.ndarray, camat: np.ndarray):
    """
    Calculate the camera azimuth, elevation, and distance from the given camera position.

    Args:
        lookat (np.ndarray): The point the camera is looking at, given as a 3D vector [x, y, z].
        camat (np.ndarray): The camera position as a 3D vector [x, y, z].

    Returns:
        tuple: A tuple containing the azimuth (degrees), elevation (degrees), and distance.
    """
    # Calculate the vector from the lookat point to the camera position
    delta = lookat - camat

    # Calculate the distance
    distance = np.linalg.norm(delta)
    distance = 1e-3 if distance < 1e-3 else distance

    # Calculate the azimuth angle in degrees
    azimuth = np.degrees(np.arctan2(delta[1], delta[0]))

    # Calculate the elevation angle in degrees
    elevation = np.degrees(np.arcsin(delta[2] / distance))

    return azimuth, elevation, distance


def preview(XML_path, render_mode="onscreen", duration=2):
    XML_path = os.path.join(os.path.dirname(__file__), XML_path)
    width = 384
    height = 360
    mj_model = mujoco.MjModel.from_xml_path(XML_path)
    mj_data = mujoco.MjData(mj_model)


    if render_mode == "onscreen":
        window = viewer.launch_passive(
            mj_model,
            mj_data,
            show_left_ui=False,
            show_right_ui=False,
        )
        scene_option = window.opt
        cam = window.cam

    elif render_mode=="offscreen":
        renderer = mujoco.Renderer(mj_model, height=height, width=width)
        scene_option = mujoco.MjvOption()
        cam = mujoco.MjvCamera()
        rgb_frames = []

    # Update scene
    scene_option.geomgroup[3] = 1

    # Update cam
    cam.elevation = -20
    cam.lookat=[0, 0, 0.04]

    # Update simulation
    tt = 0
    nstep = int(duration//mj_model.opt.timestep)
    nskip = int((1.0/30)//mj_model.opt.timestep)

    for istep in range(nstep):

        tt = mj_data.time

        cam.azimuth = 90 + 360*tt/duration # trick to rotate camera for 360 videos

        for gid in range(1, mj_model.ngeom):
            geom = mj_model.geom(gid)
            if geom.group == 3:
                geom.rgba[3] = 1 if tt>duration/2 else 0
            else:
                geom.rgba[3] = 1 if tt<duration/2 else 0


        # if tt>duration:
        #     mujoco.mj_resetData(mj_model, mj_data)
        #     time.sleep(.1)
        #     break

        mujoco.mj_step(mj_model, mj_data)
        if render_mode == "onscreen":
            window.sync()
            time.sleep(mj_model.opt.timestep)

        elif render_mode=="offscreen":
            if istep % nskip == 0:
                renderer.update_scene(mj_data, camera=cam, scene_option=scene_option)
                rgb_frames.append(renderer.render())


    if render_mode == "onscreen":
        window.close()
    else:
        file_name =  f"videos/{XML_path.split('/')[-2]}.mp4"
        print(f"saving: {file_name}")

        # check if the platform is OS -- make it compatible with quicktime
        if platform == "darwin":
            skvideo.io.vwrite(file_name, np.asarray(rgb_frames),outputdict={"-pix_fmt": "yuv420p"})
        else:
            skvideo.io.vwrite(file_name, np.asarray(rgb_frames), outputdict={"-r": str(500)})


def list_folders_in_path(path):
    """
    List all folders in the specified directory path.

    :param path: The directory path to search for folders.
    :return: A list of folder names.
    """
    try:
        # Get a list of all entries in the directory
        entries = os.listdir(path)

        # Filter out entries that are directories
        folders = [entry for entry in entries if os.path.isdir(os.path.join(path, entry))]

        return folders
    except FileNotFoundError:
        print(f"The path {path} does not exist.")
        return []
    except PermissionError:
        print(f"Permission denied for accessing the path {path}.")
        return []


@click.command(help="Preview Objects")
@click.option('-o', '--object_name', type=str, default=None)
@click.option('-r', '--render_mode', type=click.Choice(['onscreen', 'offscreen']), default="onscreen")
@click.option('-d', '--duration', type=float, default=2.0)
def preview_objects(object_name, render_mode, duration):
    """
    Preview Objects
    """

    tStart = time.time()
    if object_name:
        preview(f"{object_name}/object.xml", render_mode=render_mode, duration=duration)
    else:
        # find list of all the folder in the specified path
        path_to_search = os.path.dirname(__file__)
        folders = list_folders_in_path(path_to_search)

        for object_name in folders:
            if object_name != "google_objects":
                preview(f"{object_name}/object.xml", render_mode=render_mode, duration=duration)
                if render_mode=="onscreen":
                    print(object_name)
                    time.sleep(.1)
    print(f"Total time spent:{time.time()-tStart}", flush=True)


if __name__ == "__main__":
    preview_objects()