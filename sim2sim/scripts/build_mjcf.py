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
HEADLIGHT_AMBIENT = "0.35 0.35 0.35"
HEADLIGHT_DIFFUSE = "0.45 0.45 0.45"
HEADLIGHT_SPECULAR = "0.1 0.1 0.1"
TOP_LIGHT_HEIGHT = "1.5"
ROBOT_COLLISION_MASK = {"contype": "2", "conaffinity": "1"}

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
    # Purpose: 从 path 表示的 URDF 文件路径中读取机器人结构，并提取生成 MuJoCo 模型所需的连杆和关节信息。
    # Inputs: path 表示训练阶段使用的 URDF 模型文件位置，本函数会从这个 XML 文件中读取 link 与 joint 节点。
    # Outputs: 返回 links 和 joints；links 表示按连杆名称索引的视觉网格与惯性参数，joints 表示按 URDF 顺序保存的关节连接、位姿、转轴、限位和力矩上限。
    """Return (links, joints) where each entry keeps only what the MJCF needs."""
    root = ET.parse(path).getroot()
    links = {}
    # 提取每个 link 节点描述的连杆信息：视觉网格用于后续注册 MuJoCo 资源，惯性参数用于后续给 MuJoCo body 添加质量和转动惯量。
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
    # 提取每个 joint 节点描述的运动链信息：父子连杆关系用于重建树形结构，关节原点、转轴、限位和力矩上限用于创建 MuJoCo hinge joint 与 motor actuator。
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
            group="0", rgba="0.6 0.6 0.6 0", **ROBOT_COLLISION_MASK,
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
            **ROBOT_COLLISION_MASK,
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
            **ROBOT_COLLISION_MASK,
        )


def build():
    # Purpose: 生成 MuJoCo 可加载的 MJCF 模型文件，把训练阶段使用的 URDF 机器人描述转换成仿真部署所需的 XML 结构。
    links, joints = parse_urdf(URDF_PATH)
    children = {}
    # 根据 joints 表示的关节列表建立父连杆到子关节的索引，方便后面从 base_link 开始递归恢复整棵机器人运动链。
    for j in joints:
        children.setdefault(j["parent"], []).append(j)

    # 创建 MuJoCo XML 的根节点，并写入编译选项和全局仿真参数；这些设置决定角度单位、网格查找目录、积分步长和重力方向。
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
    visual = ET.SubElement(mujoco, "visual")
    ET.SubElement(
        visual,
        "headlight",
        ambient=HEADLIGHT_AMBIENT,
        diffuse=HEADLIGHT_DIFFUSE,
        specular=HEADLIGHT_SPECULAR,
        active="1",
    )

    # 在 asset 节点中注册视觉网格、地面纹理和地面材质；注册后的网格名称会被后续 body 里的几何元素引用。
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

    # 创建 worldbody 表示的 MuJoCo 世界主体，并加入光源和地面；地面是机器人轮子和其他碰撞几何发生接触的平面。
    worldbody = ET.SubElement(mujoco, "worldbody")
    ET.SubElement(
        worldbody, "geom", name="floor", type="plane", size="0 0 0.05",
        material="grid", friction="1.0 0.005 0.0001",
    )

    # 创建 base_link 表示的机器人基座，并添加 freejoint 表示的六自由度浮动关节，使机器人可以在 MuJoCo 中移动、倾倒和保持平衡。
    # base_link as floating body. Init height 0.35 to match cfg.init_state.pos.
    base = ET.SubElement(
        worldbody, "body", name="base_link", pos="0 0 0.35",
    )
    ET.SubElement(base, "freejoint", name="floating_base")
    add_inertial(base, links["base_link"]["inertial"])
    add_link_geoms(base, "base_link", links["base_link"])
    # A tracking spotlight stays above base_link in world coordinates and keeps
    # pointing downward, so robot roll/pitch does not rotate the light cone.
    ET.SubElement(
        base,
        "light",
        name="base_top_spotlight",
        mode="track",
        directional="false",
        castshadow="true",
        pos=f"0 0 {TOP_LIGHT_HEIGHT}",
        dir="0 0 -1",
        cutoff="45",
        exponent="10",
        ambient="0.05 0.05 0.05",
        diffuse="0.9 0.9 0.85",
        specular="0.2 0.2 0.2",
    )

    # attach 会把 parent_link 表示的父连杆下方所有子连杆递归挂到 parent_el 表示的 MuJoCo body 元素下，从而复现 URDF 中的树形运动链。
    # Recursively attach child bodies following the URDF kinematic tree.
    def attach(parent_el, parent_link):
        for j in children.get(parent_link, []):
            # 为当前子连杆创建 MuJoCo body，并用关节原点的平移和旋转描述它相对父连杆的位置关系。
            body = ET.SubElement(
                parent_el, "body", name=j["child"], pos=j["pos"], euler=j["rpy"],
            )
            # 为当前子连杆创建单自由度转动关节；关节轴来自 URDF 中的 axis 字段，角度范围来自 URDF 中的 limit 字段。
            jnt_kwargs = dict(
                name=j["name"], type="hinge", axis=j["axis"], pos="0 0 0",
            )
            if j["lower"] is not None and j["upper"] is not None:
                jnt_kwargs["range"] = f"{j['lower']} {j['upper']}"
                jnt_kwargs["limited"] = "true"
            else:
                jnt_kwargs["limited"] = "false"
            ET.SubElement(body, "joint", **jnt_kwargs)
            # 给当前子连杆添加惯性和几何元素；其中几何元素会根据连杆类型选择视觉网格、碰撞网格、轮子圆柱或基座盒子。
            add_inertial(body, links[j["child"]]["inertial"])
            add_link_geoms(body, j["child"], links[j["child"]])
            attach(body, j["child"])

    attach(base, "base_link")

    # 为每个真实关节创建 motor actuator 表示的力矩输入通道，使 Python 控制器可以通过 MuJoCo 的控制数组向对应关节施加力矩。
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

    # 将内存中的 XML 树转换成带缩进的文本，并写入 OUT_PATH 表示的 MJCF 输出文件路径。
    rough = ET.tostring(mujoco, encoding="utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(pretty)
    print("wrote", OUT_PATH)


if __name__ == "__main__":
    build()
