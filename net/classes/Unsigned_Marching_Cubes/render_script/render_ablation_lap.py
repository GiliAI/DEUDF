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
sys.path.append('D:\\BlenderToolbox\\') # change this to your path to “path/to/BlenderToolbox/
import BlenderToolBox as bt
import bmesh
import datetime
from math import sqrt, radians, pi, cos, sin
from mathutils import Vector, Matrix
from random import random, seed, uniform, randint, randrange
from enum import IntEnum
from colorsys import hls_to_rgb
import shutil
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


def create_materials():
    ret = []
    edgeThickness = 0.005
    edgeColor = colorObj((0,0,0,0),0.5, 1.0, 1.0, 0.0, 0.0)
    meshColor = (179.0/255, 190.0/255, 203.0/255, 1)
    AOStrength = 1.0
    hull_base_color = (0.0,0.0,0.9, 1.0)
    for material in Material:
        mat = bpy.data.materials.new(name=material.name)
        mat.use_nodes = True
        shader_node = getShaderNode(mat)
        shader_node.inputs["Metallic"].default_value = 0.2
        shader_node.inputs["Specular"].default_value = 0.75
        shader_node.inputs["Base Color"].default_value = hull_base_color
        shader_node.inputs["Specular Tint"].default_value = 1.0
        shader_node.inputs["Clearcoat Roughness"].default_value = 0.03
        shader_node.inputs["Roughness"].default_value = 0.4
        shader_node.inputs["Alpha"].default_value = 1.0
        shader_node.inputs["Sheen Tint"].default_value = 0.0
        shader_node.inputs["Emission Strength"].default_value = 0.0
        tree = mat.node_tree

#        # set principled BSDF
#        tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = 0.7
#        tree.nodes["Principled BSDF"].inputs['Sheen Tint'].default_value = 0

        # add Ambient Occlusion
        tree.nodes.new('ShaderNodeAmbientOcclusion')
        tree.nodes.new('ShaderNodeGamma')
        tree.nodes.new('ShaderNodeMixRGB')
        tree.nodes["Mix"].blend_type = 'MULTIPLY'
        tree.nodes["Gamma"].inputs["Gamma"].default_value = AOStrength
        tree.nodes["Gamma"].location.x -= 600
        tree.nodes["Ambient Occlusion"].inputs["Distance"].default_value = 10.0
        tree.nodes["Ambient Occlusion"].inputs["Color"].default_value = meshColor
        tree.links.new(tree.nodes["Ambient Occlusion"].outputs['Color'], tree.nodes['Mix'].inputs['Color1'])
        tree.links.new(tree.nodes["Ambient Occlusion"].outputs['AO'], tree.nodes['Gamma'].inputs['Color'])
        tree.links.new(tree.nodes["Gamma"].outputs['Color'], tree.nodes['Mix'].inputs['Color2'])
        tree.links.new(tree.nodes["Mix"].outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])

        # add edge wireframe
        tree.nodes.new(type="ShaderNodeWireframe")
        wire = tree.nodes[-1]
        wire.inputs[0].default_value = edgeThickness
        wire.location.x -= 200
        wire.location.y -= 200
        tree.nodes.new(type="ShaderNodeBsdfDiffuse")
        mat_wire = tree.nodes[-1]
        HSVNode = tree.nodes.new('ShaderNodeHueSaturation')
        HSVNode.inputs['Color'].default_value = edgeColor.RGBA
        HSVNode.inputs['Saturation'].default_value = edgeColor.S
        HSVNode.inputs['Value'].default_value = edgeColor.V
        HSVNode.inputs['Hue'].default_value = edgeColor.H
        HSVNode.location.x -= 200
        # set color brightness/contrast
        BCNode = tree.nodes.new('ShaderNodeBrightContrast')
        BCNode.inputs['Bright'].default_value = edgeColor.B
        BCNode.inputs['Contrast'].default_value = edgeColor.C
        BCNode.location.x -= 400

        tree.links.new(HSVNode.outputs['Color'],BCNode.inputs['Color'])
        tree.links.new(BCNode.outputs['Color'],mat_wire.inputs['Color'])

        tree.nodes.new('ShaderNodeMixShader')
        tree.links.new(wire.outputs[0], tree.nodes['Mix Shader'].inputs[0])
        tree.links.new(mat_wire.outputs['BSDF'], tree.nodes['Mix Shader'].inputs[2])
        tree.links.new(tree.nodes["Principled BSDF"].outputs['BSDF'], tree.nodes['Mix Shader'].inputs[1])
        tree.links.new(tree.nodes["Mix Shader"].outputs['Shader'], tree.nodes['Material Output'].inputs['Surface'])


        ret.append(mat)

    # Choose a base color for the spaceship hull


    # Build the hull texture
    mat = ret[Material.hull]
#    set_mat_basics(mat, hull_base_color)

    return ret


def generate_model(file_dir,file_name,assign_materials=True):

    #将.ply模型添加到场景内
    scene = bpy.context.scene
    me=bpy.ops.import_mesh.ply(filepath=file_dir+file_name)
    obj = bpy.data.objects[file_name[0:-4]]

    # 将模型设置为激活状态
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    #调整模型的位置参数（location,rotation,scale）
    #bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
    ob = bpy.context.object
    ob.rotation_euler=(radians(90),radians(0),radians(270))
    ob.scale = (0.5, 0.5, 0.5)
    # ob.rotation_euler=(radians(-128),radians(0),radians(180))
    ob.location=(-0.0      ,-0.0      ,-ob.bound_box[0][1]*0.5       )
    # ob.scale=(40,40.0,40.0)
    # print(ob.bound_box[0][0])
    # print(ob.bound_box[0][1])
    # print(ob.bound_box[0][2])
    # 为模型添加材质
    edgeThickness = 0.0005
    edgeColor = bt.colorObj((0, 0, 0, 0), 0.5, 1.0, 1.0, 0.0, 0.0)
    meshRGBA = (179.0/255, 190.0/255, 203.0/255, 1)
    AOStrength = 1.0
    bt.setMat_edge(ob, edgeThickness, edgeColor, meshRGBA, AOStrength)

    return obj


def set_render_properties(engineName):

    #设置渲染引擎
    if engineName=='CYCLES':
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.view_layer.cycles.use_denoising=True#设置降噪
    else:
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'#设置渲染器为EEVEE
    bpy.context.scene.view_settings.view_transform = 'Standard'#设置：渲染属性->色彩管理->查看变换->标准

def set_output_properties(res_x,res_y,fov):
    #设置输出图像尺寸
    bpy.context.scene.render.resolution_x = res_x
    bpy.context.scene.render.resolution_y = res_y
    #设置输出图片压缩率
    bpy.context.scene.render.image_settings.compression = 0
    # bpy.context.scene.camera.rotation_mode = 'XYZ'
    # bpy.context.scene.camera.data.angle = radians(fov)
#def set_world_properties():


def set_light_properties():
    bpy.context.object.data.energy = 20#光照强度
    bpy.context.object.data.shadow_soft_size = 1#电源半径
    bpy.context.object.data.cycles.max_bounces = 512#最多反弹次数
    bpy.context.object.data.cycles.cast_shadow = True#投射阴影
    bpy.context.object.data.cycles.use_multiple_importance_sampling = True#多重重要性采样




if __name__ == "__main__":
    in_floder = "D:\\render_data\\all_data_for_render\\ablation\\"
    # del_file("D:\\blender_outs")
    for obj_name in os.listdir(in_floder):
        print(obj_name)
        reset_scene()
        bpy.context.scene.render.film_transparent = True


        # Settings
        output_path = '' #输出路径
        fov = 60 # degrees
        fps = 30
        res_x = 1920*2
        res_y = 1080*2
        numSamples = 100
        exposure = 1.5
        bt.blenderInit(res_x, res_y, numSamples, exposure)
        obj = generate_model(file_dir=in_floder, file_name=obj_name)
        bt.invisibleGround(shadowBrightness=0.9)
        ## set light
        lightAngle = (290.318, -1.82584, -48.9597)
        strength = 3
        shadowSoftness = 0.3
        sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)

        bpy.ops.object.light_add(type='POINT')
        light_obj=bpy.context.object
        light_obj.name='Light1'
        light_obj.location=(0 ,0,0.7 )
        set_light_properties()

        bpy.ops.object.light_add(type='POINT')
        light_obj=bpy.context.object
        light_obj.name='Light2'
        light_obj.location=(0 ,0,0.3)
        set_light_properties()

        # bt.invisibleGround(location=(0,0,-0.178),shadowBrightness=0.9)
        ## set ambient light
        bt.setLight_ambient(color=(0.1, 0.1, 0.1, 1))

        ## set gray shadow to completely white with a threshold
        # bt.shadowThreshold(alphaThreshold=0.05, interpolationMode='CARDINAL')

        #add camera
        # bpy.ops.object.camera_add()
        # camera_obj=bpy.context.object
        # bpy.context.scene.camera=camera_obj
        # camera_obj.name="Camera"
        # R = 4.5  # radius of the circle camera path
        # H = 1  # height of the circle camera path
        # lookAtPos = (0, 0, 0.1)  # look at position
        # R = 4  # radius of the circle camera path
        # H = 0.9  # height of the circle camera path
        camLocation = (0.919265  ,-1.035353   ,1.277597 )
        lookAtLocation = (0, 0.0, 0.4)
        focalLength = 45  # (UI: click camera > Object Data > Focal Length)
        cam = bt.setCamera(camLocation, lookAtLocation, focalLength)

        # Generate a new model

        set_render_properties('CYCLES')
        folder = "D:\\blender_outs\\{}".format(obj_name[:-4])
        os.makedirs(folder, exist_ok=True)
        # del_file(folder)
        # Batch render the movie frames
        inv_fps = 1/float(fps)
        movie_duration = 0
        scene = bpy.data.scenes["Scene"]
        # set_output_properties(res_x,res_y,fov)
        bpy.ops.wm.save_mainfile(filepath=os.getcwd() + '/test.blend')
        outputPath = os.path.join(folder, obj_name[:-4] + "_" + in_floder.split("\\")[-2] + '_.png')
        # bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        bt.renderImage(outputPath, cam)
        # frame = 0
        # timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        # while movie_duration < total_movie_duration:
        #
        #     # Position and orient the camera
        #     rad = radians(yaw_offset + (yaw_rate * movie_duration))
        #     scene.camera.rotation_euler = (radians(60), 0, rad)
        #     scene.camera.location = (sin(rad) * camera_pole_length,
        #                              cos(rad) * -camera_pole_length,
        #                              cos(radians(60))*camera_pole_length)
        #     # Render the scene to disk
        #     print(in_floder.split("\\"))
        #     filename = obj_name[:-4] +"_" + str(frame).zfill(3) + "_" + in_floder.split("\\")[-2]+ '.png'
        #     bpy.data.scenes['Scene'].render.filepath = os.path.join(folder, filename)
        #     # print('Rendering frame ' + str(frame) + '...')
        #     bpy.ops.render.render(write_still=True)
        #     frame += 1
        #     movie_duration += inv_fps
