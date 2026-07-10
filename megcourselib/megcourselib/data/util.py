from os import path


def get_scene_path(scene_id, stim_root, id_str_len=12, strict=True):
    # make sure scene_id is an integer
    scene_id = int(scene_id)
    # now convert to str
    scene_str = str(scene_id)
    # pad with zeros to match filename
    scene_str = scene_str.zfill(id_str_len)
    # assemble filename
    fname = f"{scene_str}_MEG_size.jpg"
    # assemble path
    fpath = path.join(stim_root, fname)
    if strict:
        assert path.exists(fpath), f"{fpath} does not exist."
    return fpath
