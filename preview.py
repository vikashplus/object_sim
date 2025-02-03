import mujoco
import time

from mujoco import viewer
import os

def preview(XML_path):
  XML_path = os.path.join(os.path.dirname(__file__), XML_path)
  render = "onscreen"
  # render = "offscreen"
  width = 400
  height = 400
  mj_model = mujoco.MjModel.from_xml_path(XML_path)
  mj_data = mujoco.MjData(mj_model)


  if render == "onscreen":
    window = viewer.launch_passive(
        mj_model,
        mj_data,
        show_left_ui=False,
        show_right_ui=False,
    )
  elif render=="offscreen":
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    scene_option = mujoco.MjvOption()
    import ipdb; ipdb.set_trace()


  tmax = 2
  window.cam.elevation = -20
  window.opt.geomgroup[3] = 1


  # Update simulation
  while window.is_running():

    tt = mj_data.time

    window.cam.azimuth = 90 + 360*tt/tmax # trick to rotate camera for 360 videos
    # print(tt, window.cam.azimuth)

    for gid in range(1, mj_model.ngeom):
      geom = mj_model.geom(gid)
      if geom.group == 3:
        # geom.rgba[3] = min(tt/tmax, 1)
        geom.rgba[3] = 1 if tt>tmax/2 else 0
      else:
        # geom.rgba[3] = max(1-tt/tmax, 0)
        geom.rgba[3] = 1 if tt<tmax/2 else 0


    if tt>tmax:
      mujoco.mj_resetData(mj_model, mj_data)
      time.sleep(.1)
      break

    mujoco.mj_step(mj_model, mj_data)
    window.sync()
    time.sleep(mj_model.opt.timestep)
  window.close()


import os

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



import os
import click

@click.command(help="Preview Objects")
@click.option('-o', '--object_name', type=str, default=None)
def preview_objects(object_name):
    """
    Preview Objects
    """

    if object_name:
        preview(f"{object_name}/object.xml")
    else:
        # find list of all the folder in the specified path
        path_to_search = os.path.dirname(__file__)
        folders = list_folders_in_path(path_to_search)

        for object_name in folders:
            if object_name != "google_objects":
                print(object_name)
                preview(f"{object_name}/object.xml")
                time.sleep(.1)



if __name__ == "__main__":
    preview_objects()