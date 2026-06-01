"""Generate a MuJoCo MJCF model for the FYT wheel-legged robot from its URDF.

The robot is an open-chain biped with two legs (thigh + leg + wheel each).
We parse the URDF joints/links directly so every transform, inertia and limit
matches the asset that was used during Isaac Gym training; nothing is hand
copied. The only deliberate modelling choices for MuJoCo are:

- a floating base (freejoint) so the robot can fall / balance;
- wheel collision is a cylinder primitive sized from the wheel mesh bounds
  (radius 0.0579 m, half-width 0.019 m) for stable, cheap wheel-ground contact;
- thigh / leg / base keep mesh collision (matching the training asset);
- six torque "motor" actuators, because the VMC + PD control law runs in Python
  and feeds joint torques straight into the simulator.

Run once to (re)create sim2sim/model/wheel_legged_v4.xml.
"""
import os
import xml.etree.ElementTree as ET

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
URDF_PATH = os.path.join(
    REPO_ROOT, "resources/robots/wheel_legged_fyt/urdf/wheel_legged_v4_isaac.urdf"
)
MESH_DIR = os.path.join(REPO_ROOT, "resources/robots/wheel_legged_fyt/meshes")
OUT_PATH = os.path.join(REPO_ROOT, "sim2sim/model/wheel_legged_v4.xml")

# Wheel collision cylinder, measured from left_wheel_link.STL bounds.
WHEEL_RADIUS = 0.0579
WHEEL_HALF_WIDTH = 0.019
WHEEL_LINKS = {"left_wheel_link", "right_wheel_link"}

# base_link.STL was decimated in place (via MeshLab) to 190k faces, now within
# MuJoCo's 200k-face cap, so its real shape loads directly through meshdir like
# every other link -- no separate decimated copy is needed.
# COLLISION still uses a box primitive: it is never rendered, self-collision is
# off in training, and a convex hull of the real mesh would swallow the leg
# joints. The box is sized/centred from the measured mesh bounds
# (0.60 x 0.39 x 0.47, centroid z=0.215 relative to the link origin).
BASE_BOX_HALF = "0.30 0.196 0.236"
BASE_BOX_POS = "0 0 0.215"


def parse_urdf(path):
    """Return (links, joints) where each entry keeps only what the MJCF needs."""
    root = ET.parse(path).getroot()
    links = {}
    for link in root.findall("link"):
        name = link.get("name")
        inertial = link.find("inertial")
        info = {"mesh": None, "inertial": None}
        visual = link.find("visual/geometry/mesh")
        if visual is not None:
            info["mesh"] = os.path.basename(visual.get("filename"))
        if inertial is not None:
            origin = inertial.find("origin")
            inertia = inertial.find("inertia")
            info["inertial"] = {
                "pos": origin.get("xyz", "0 0 0"),
                "mass": inertial.find("mass").get("value"),
                "ixx": inertia.get("ixx"),
                "iyy": inertia.get("iyy"),
                "izz": inertia.get("izz"),
                "ixy": inertia.get("ixy"),
                "ixz": inertia.get("ixz"),
                "iyz": inertia.get("iyz"),
            }
        links[name] = info

    joints = []
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        limit = joint.find("limit")
        joints.append(
            {
                "name": joint.get("name"),
                "parent": joint.find("parent").get("link"),
                "child": joint.find("child").get("link"),
                "pos": origin.get("xyz", "0 0 0"),
                "rpy": origin.get("rpy", "0 0 0"),
                "axis": joint.find("axis").get("xyz", "0 0 1"),
                "lower": limit.get("lower") if limit is not None else None,
                "upper": limit.get("upper") if limit is not None else None,
                "effort": limit.get("effort") if limit is not None else None,
            }
        )
    return links, joints


def fullinertia(inertial):
    # MuJoCo fullinertia order: ixx iyy izz ixy ixz iyz
    return "{ixx} {iyy} {izz} {ixy} {ixz} {iyz}".format(**inertial)


def add_inertial(body_el, inertial):
    ET.SubElement(
        body_el,
        "inertial",
        pos=inertial["pos"],
        mass=inertial["mass"],
        fullinertia=fullinertia(inertial),
    )


def add_link_geoms(body_el, link_name, link_info):
    """Visual mesh geom + collision geom for one link."""
    mesh_name = link_name  # mesh asset registered under the link name
    if link_name == "base_link":
        # Visual: real decimated chassis mesh (origin matches URDF visual at 0).
        ET.SubElement(
            body_el, "geom", type="mesh", mesh="base_link",
            contype="0", conaffinity="0", group="1", rgba="0.75 0.75 0.78 1",
        )
        # Collision: invisible box (not rendered, only for base-ground contact).
        ET.SubElement(
            body_el, "geom", type="box", size=BASE_BOX_HALF, pos=BASE_BOX_POS,
            group="0", rgba="0.6 0.6 0.6 0",
        )
        return
    if link_info["mesh"] is not None:
        # Visual only (contype/conaffinity 0): never collides, just rendered.
        ET.SubElement(
            body_el,
            "geom",
            type="mesh",
            mesh=mesh_name,
            contype="0",
            conaffinity="0",
            group="1",
            rgba="0.75 0.75 0.78 1",
        )
    if link_name in WHEEL_LINKS:
        # Wheel collision: cylinder along local z (the joint/spin axis).
        ET.SubElement(
            body_el,
            "geom",
            type="cylinder",
            size=f"{WHEEL_RADIUS} {WHEEL_HALF_WIDTH}",
            friction="1.0 0.005 0.0001",
            rgba="0.2 0.2 0.2 1",
        )
    elif link_info["mesh"] is not None:
        # Non-wheel links collide with their mesh, matching the training asset.
        ET.SubElement(
            body_el,
            "geom",
            type="mesh",
            mesh=mesh_name,
            group="0",
            rgba="0.6 0.6 0.6 0.4",
        )


def build():
    links, joints = parse_urdf(URDF_PATH)
    children = {}
    for j in joints:
        children.setdefault(j["parent"], []).append(j)

    mujoco = ET.Element("mujoco", model="wheel_legged_v4")
    ET.SubElement(
        mujoco,
        "compiler",
        angle="radian",
        meshdir=MESH_DIR,
        autolimits="true",
        balanceinertia="true",
    )
    ET.SubElement(mujoco, "option", timestep="0.005", integrator="Euler", gravity="0 0 -9.81")

    # Mesh assets + ground texture/material.
    asset = ET.SubElement(mujoco, "asset")
    for name, info in links.items():
        # All links (including the decimated base_link) load their STL from
        # meshdir by file name.
        if info["mesh"] is not None:
            ET.SubElement(asset, "mesh", name=name, file=info["mesh"])
    ET.SubElement(
        asset, "texture", name="grid", type="2d", builtin="checker",
        rgb1="0.2 0.3 0.4", rgb2="0.1 0.15 0.2", width="512", height="512",
    )
    ET.SubElement(
        asset, "material", name="grid", texture="grid",
        texrepeat="5 5", reflectance="0.1",
    )

    worldbody = ET.SubElement(mujoco, "worldbody")
    ET.SubElement(
        worldbody, "light", pos="0 0 3", dir="0 0 -1", diffuse="0.8 0.8 0.8",
    )
    ET.SubElement(
        worldbody, "geom", name="floor", type="plane", size="0 0 0.05",
        material="grid", friction="1.0 0.005 0.0001",
    )

    # base_link as floating body. Init height 0.35 to match cfg.init_state.pos.
    base = ET.SubElement(
        worldbody, "body", name="base_link", pos="0 0 0.35",
    )
    ET.SubElement(base, "freejoint", name="floating_base")
    add_inertial(base, links["base_link"]["inertial"])
    add_link_geoms(base, "base_link", links["base_link"])

    # Recursively attach child bodies following the URDF kinematic tree.
    def attach(parent_el, parent_link):
        for j in children.get(parent_link, []):
            body = ET.SubElement(
                parent_el, "body", name=j["child"], pos=j["pos"], euler=j["rpy"],
            )
            jnt_kwargs = dict(
                name=j["name"], type="hinge", axis=j["axis"], pos="0 0 0",
            )
            if j["lower"] is not None and j["upper"] is not None:
                jnt_kwargs["range"] = f"{j['lower']} {j['upper']}"
                jnt_kwargs["limited"] = "true"
            else:
                jnt_kwargs["limited"] = "false"
            ET.SubElement(body, "joint", **jnt_kwargs)
            add_inertial(body, links[j["child"]]["inertial"])
            add_link_geoms(body, j["child"], links[j["child"]])
            attach(body, j["child"])

    attach(base, "base_link")

    # Torque actuators: control input is the joint torque computed in Python.
    actuator = ET.SubElement(mujoco, "actuator")
    for j in joints:
        eff = float(j["effort"]) if j["effort"] is not None else 30.0
        ET.SubElement(
            actuator, "motor", name=j["name"], joint=j["name"],
            ctrlrange=f"{-eff} {eff}", ctrllimited="true", gear="1",
        )

    # minidom pretty-print keeps the XML readable (ET.indent needs Python 3.9+).
    from xml.dom import minidom

    rough = ET.tostring(mujoco, encoding="utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(pretty)
    print("wrote", OUT_PATH)


if __name__ == "__main__":
    build()
