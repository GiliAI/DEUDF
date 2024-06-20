#
# spaceship_generator.py
#
# This is a Blender script that uses procedural generation to create
# textured 3D spaceship models. Tested with Blender 2.77a.
#
# michael@spaceduststudios.com
# https://github.com/a1studmuffin/SpaceshipGenerator
#

import sys
import os
import os.path
import bpy
import sys
sys.path.append('D:\\BlenderToolbox') # change this to your path to “path/to/BlenderToolbox/
import BlenderToolBox as bt
import bmesh
import mathutils
import datetime
from math import sqrt, radians, pi, cos, sin
from mathutils import Vector, Matrix
from random import random, seed, uniform, randint, randrange
from enum import IntEnum
from colorsys import hls_to_rgb
import shutil
import json
DIR = os.path.dirname(os.path.abspath(__file__))

class colorObj(object):
    def __init__(self, RGBA, \
    H = 0.5, S = 1.0, V = 1.0,\
    B = 0.0, C = 0.0):
        self.H = H # hue
        self.S = S # saturation
        self.V = V # value
        self.RGBA = RGBA
        self.B = B # birghtness
        self.C = C # contrast


def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
            os.removedirs(c_path)
        else:
            os.remove(c_path)

def resource_path(*path_components):
    return os.path.join(DIR, *path_components)

# Deletes all existing spaceships and unused materials from the scene
def reset_scene():
    #for item in bpy.data.objects:
        #.select_set(item.name.startswith('Spaceship'))
    #bpy.ops.object.delete()
    if(len(bpy.data.objects)!=0):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
    for material in bpy.data.materials:
        if not material.users:
            bpy.data.materials.remove(material)
    for texture in bpy.data.textures:
        if not texture.users:
            bpy.data.textures.remove(texture)

class Material(IntEnum):
    hull = 0            # Plain spaceship hull

# Returns shader node
def getShaderNode(mat):
    ntree = mat.node_tree
    node_out = ntree.get_output_node('EEVEE')
    shader_node = node_out.inputs['Surface'].links[0].from_node
    return shader_node

def getShaderInput(mat, name):
    shaderNode = getShaderNode(mat)
    return shaderNode.inputs[name]

# Sets some basic properties for a hull material.
def set_mat_basics(mat, color):
    shader_node = getShaderNode(mat)
    shader_node.inputs["Metallic"].default_value = 0.2
    shader_node.inputs["Specular"].default_value = 0.75
    shader_node.inputs["Base Color"].default_value = color
    shader_node.inputs["Specular Tint"].default_value = 1.0
    shader_node.inputs["Clearcoat Roughness"].default_value = 0.03
    shader_node.inputs["Roughness"].default_value = 0.4
    shader_node.inputs["Alpha"].default_value = 1.0
    shader_node.inputs["Sheen Tint"].default_value = 0.0
    shader_node.inputs["Emission Strength"].default_value = 0.0

# Creates all our materials and returns them as a list.

# Generates a textured spaceship mesh and returns the object.
# Just uses global cube texture coordinates rather than generating UVs.
# Takes an optional random seed value to generate a specific spaceship.
# Allows overriding of some parameters that affect generation.
def generate_model(file_dir,file_prefix,iter):
    file_name =  file_prefix+".ply"
    me = bpy.ops.import_mesh.ply(filepath=file_dir + file_name)
    obj = bpy.data.objects[file_name[:-4]]

    # 将模型设置为激活状态
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # 调整模型的位置参数（location,rotation,scale）
    # bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
    ob = bpy.context.object
    ob.rotation_euler = (radians(90), radians(0), radians(0))
    # ob.scale = (1.5, 1.5, 1.5)
    # ob.rotation_euler=(radians(-128),radians(0),radians(180))
    ob.location = (-0.0, -0.0, -ob.bound_box[0][1])
    # ob.scale=(40,40.0,40.0)
    # print(ob.bound_box[0][0])
    # print(ob.bound_box[0][1])
    # print(ob.bound_box[0][2])

    # 为模型添加材质
    # bpy.ops.object.shade_smooth()
    alpha = 0.1
    transmission = 0.5
    bt.setMat_balloon(ob, bt.colorObj((0.639, 0.724, 0.796, 1), 0.5, 1.5, 1.0, 0.0, 0.0))
    # bt.setMat_transparent(ob, bt.colorObj((179.0 / 255, 190.0 / 255, 203.0 / 255, 1), 0.5, 1.5, 1.0, 0.0, 0.0),0.1,0.1)
    # meshColor = bt.colorObj((179.0 / 255, 190.0 / 255, 203.0 / 255, 1), 0.5, 1.0, 1.0, 0.0, 0.0)
    # roughness = 0.1
    # bt.setMat_glass(ob, meshColor, roughness)
    # bt.setMat_balloon(ob, bt.colorObj(bt.derekBlue, 0.5, 1.5, 1.0, 0.0, 0.0), alpha)
    # meshColor = bt.colorObj((.9, .9, .9, 1), 0.5, 1.0, 1.0, 0.0, 0.0)
    # roughness = 0.1
    # bt.setMat_glass(ob, meshColor, roughness)
    return obj

def set_render_properties(engineName):

    #设置渲染引擎
    if engineName=='CYCLES':
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.view_layer.cycles.use_denoising=True#设置降噪
        bpy.context.scene.cycles.samples = 128
    else:
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'#设置渲染器为EEVEE
    bpy.context.scene.view_settings.view_transform = 'Standard'#设置：渲染属性->色彩管理->查看变换->标准

def set_output_properties(res_x,res_y,fov):
    #设置输出图像尺寸
    bpy.context.scene.render.resolution_x = res_x
    bpy.context.scene.render.resolution_y = res_y
    #设置输出图片压缩率
    bpy.context.scene.render.image_settings.compression = 0
    bpy.context.scene.camera.rotation_mode = 'XYZ'
    bpy.context.scene.camera.data.angle = radians(fov)
#def set_world_properties():


def set_light_properties():
    bpy.context.object.data.energy = 10000#光照强度
    bpy.context.object.data.shadow_soft_size = 5#电源半径
    bpy.context.object.data.cycles.max_bounces = 1024#最多反弹次数
    bpy.context.object.data.cycles.cast_shadow = True#投射阴影
    bpy.context.object.data.cycles.use_multiple_importance_sampling = True#多重重要性采样



if __name__ == "__main__":
    in_floder = "D:\\df3d\\"
    with open("D:\\Projects\\CXH\\Unsigned_Marching_Cubes\\batch_file\\cloth_sub_1-3.json", "r") as f:
        paths = json.load(f)
    print(paths)
    for object in paths:

        # del_file("D:\\blender_outs")
        obj_name=object[:-4]
        print(obj_name)
        reset_scene()
        # bpy.context.scene.render.film_transparent = False
        # Settings
        output_path = '' #输出路径
        total_movie_duration = 8 #视频总持续时长
        yaw_rate = 45 # degrees/sec
        yaw_offset = 40 # degrees/sec
        camera_pole_length = 3
        fov = 60 # degrees
        fps = 30
        res_x = 1920//2
        res_y = 1080//2

        bt.invisibleGround(shadowBrightness=0.1)
        #set light properties
        #add light
        lightAngle = (21.7, 8.62, -131)
        strength = 0.5
        shadowSoftness = 0.3
        sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)
        bt.setLight_ambient(color=(1, 1, 1, 1),strength=1)

        #add camera
        bpy.ops.object.camera_add()
        camera_obj=bpy.context.object
        bpy.context.scene.camera=camera_obj
        camera_obj.name="Camera"

        # Generate a new model

        set_render_properties('CYCLES')
        # bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        # bpy.context.preferences.addons['cycles'].preferences.compute_device = 'CUDA_MULTI_0'
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
        folder = "D:\\blender_outs\\film\\{}\\".format(obj_name)
        os.makedirs(folder, exist_ok=True)
        # del_file(folder)
        # Batch render the movie frames
        inv_fps = 1/float(fps)
        movie_duration = 0
        scene = bpy.data.scenes["Scene"]
        set_output_properties(res_x,res_y,fov)
        frame = 0
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        lookAtLocation = (0,0,0.3)
        obj = generate_model(file_dir=in_floder, file_prefix=obj_name, iter=frame)
        while movie_duration<total_movie_duration:
            rad = radians(yaw_offset + (yaw_rate * movie_duration))
            obj_rad = 180+yaw_rate * movie_duration
            # obj = bpy.data.objects["mesh_0"]

            # 将模型设置为激活状态
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)

            # 调整模型的位置参数（location,rotation,scale）
            # bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
            ob = bpy.context.object
            ob.rotation_euler = (radians(90), radians(0), radians(180+obj_rad))
            # Position and orient the camera
            rad = radians(yaw_offset + (yaw_rate * movie_duration))
            if frame == 0:
                scene.camera.location = (sin(rad) * camera_pole_length,
                                         cos(rad) * -camera_pole_length,
                                         0.8)
            if frame == 1:
                bpy.ops.wm.save_mainfile(filepath=os.getcwd() + '/test.blend')
            loc = mathutils.Vector(lookAtLocation)
            direction = loc - scene.camera.location
            rotQuat = direction.to_track_quat('-Z', 'Y')
            scene.camera.rotation_euler = rotQuat.to_euler()

            # Render the scene to disk
            filename = obj_name +"_" + str(frame).zfill(4)+ '.png'
            bpy.data.scenes['Scene'].render.filepath = os.path.join(folder, filename)
            bpy.ops.render.render(write_still=True)
            # print('Rendering frame ' + str(frame) + '...')
            frame += 1
            movie_duration += inv_fps

            bpy.ops.object.select_all(action='DESELECT')
