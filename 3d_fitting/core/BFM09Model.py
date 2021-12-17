import torch
import torch.nn as nn
import numpy as np
from pytorch3d.structures import Meshes
from core.BaseModel import BaseReconModel
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    blending
)
from pytorch3d.ops import interpolate_face_attributes

import pytorch3d.io as io

class BFM09ReconModel(BaseReconModel):
    def __init__(self, model_dict, **kargs):
        super(BFM09ReconModel, self).__init__(**kargs)

        self.skinmask = torch.tensor(
            model_dict['skinmask'], requires_grad=False, device=self.device)

        print('skinmask',self.skinmask.shape)

        self.kp_inds = torch.tensor(
            model_dict['keypoints']-1).squeeze().long().to(self.device)


        print('kp_inds',self.kp_inds.shape)


        self.meanshape = torch.tensor(model_dict['meanshape'],
                                      dtype=torch.float32, requires_grad=False,
                                      device=self.device)


        print('meanshape',self.meanshape.shape)


        self.idBase = torch.tensor(model_dict['idBase'],
                                   dtype=torch.float32, requires_grad=False,
                                   device=self.device)


        print('idBase',self.idBase.shape)



        self.expBase = torch.tensor(model_dict['exBase'],
                                    dtype=torch.float32, requires_grad=False,
                                    device=self.device)


        print('expBase',self.expBase.shape)


        self.meantex = torch.tensor(model_dict['meantex'],
                                    dtype=torch.float32, requires_grad=False,
                                    device=self.device)


        print('meantex',self.meantex.shape)


        self.texBase = torch.tensor(model_dict['texBase'],
                                    dtype=torch.float32, requires_grad=False,
                                    device=self.device)


        print('texBase',self.texBase.shape)


        self.tri = torch.tensor(model_dict['tri']-1,
                                dtype=torch.int64, requires_grad=False,
                                device=self.device)

        print('tri',self.tri.shape)


        self.point_buf = torch.tensor(model_dict['point_buf']-1,
                                      dtype=torch.int64, requires_grad=False,
                                      device=self.device)
        print('point_buf',self.point_buf.shape)

    def get_lms(self, vs):
        lms = vs[:, self.kp_inds, :]
        return lms

    def split_coeffs(self, coeffs):
        id_coeff = coeffs[:, :80]  # identity(shape) coeff of dim 80
        exp_coeff = coeffs[:, 80:144]  # expression coeff of dim 64
        tex_coeff = coeffs[:, 144:224]  # texture(albedo) coeff of dim 80
        # ruler angles(x,y,z) for rotation of dim 3
        angles = coeffs[:, 224:227]
        # lighting coeff for 3 channel SH function of dim 27
        gamma = coeffs[:, 227:254]
        translation = coeffs[:, 254:]  # translation coeff of dim 3

        return id_coeff, exp_coeff, tex_coeff, angles, gamma, translation

    def merge_coeffs(self, id_coeff, exp_coeff, tex_coeff, angles, gamma, translation):
        coeffs = torch.cat([id_coeff, exp_coeff, tex_coeff,
                            angles, gamma, translation], dim=1)
        return coeffs

    def forward(self, coeffs, render=True):
        batch_num = coeffs.shape[0]

        id_coeff, exp_coeff, tex_coeff, angles, gamma, translation = self.split_coeffs(coeffs)

        vs = self.get_vs(id_coeff, exp_coeff)

        rotation = self.compute_rotation_matrix(angles)

        vs_t = self.rigid_transform(vs, rotation, translation)

        lms_t = self.get_lms(vs_t)

        lms_proj = self.project_vs(lms_t)
        
        lms_proj = torch.stack([lms_proj[:, :, 0], self.img_size-lms_proj[:, :, 1]], dim=2)


        if render:
            face_texture = self.get_color(tex_coeff)

            face_norm = self.compute_norm(vs, self.tri, self.point_buf)
            
            face_norm_r = face_norm.bmm(rotation)

            face_color = self.add_illumination(face_texture, face_norm_r, gamma)

            face_color_tv = TexturesVertex(face_color)
            

            mesh = Meshes(vs_t, self.tri.repeat(batch_num, 1, 1), face_color_tv)

            rendered_img = self.renderer(mesh)

            rendered_img = torch.clamp(rendered_img, 0, 255)

            #fragments = self.rasterizer(mesh)

            # packing_list = [i[j] for i, j in zip(mesh.textures.verts_uvs_list(), mesh.textures.faces_uvs_list())]

            # faces_verts_uvs = torch.cat(packing_list)

            # pixel_uvs = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, face_color)

            # N, H_out, W_out, K = fragments.pix_to_face.shape

            # # pixel_uvs: (N, H, W, K, 2) -> (N, K, H, W, 2) -> (NK, H, W, 2)
            # pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4).reshape(N * K, H_out, W_out, 2)

            # pixel_uvs = pixel_uvs * 2.0 - 1.0


            return {'rendered_img': rendered_img,
                    'lms_proj': lms_proj,
                    'face_texture': face_texture,
                    'vs': vs_t,
                    'tri': self.tri,
                    'color': face_color}
        else:
            return {'lms_proj': lms_proj}



    def get_vs(self, id_coeff, exp_coeff):
        
        n_b = id_coeff.size(0)

        face_shape = torch.einsum('ij,aj->ai', self.idBase, id_coeff) + \
            torch.einsum('ij,aj->ai', self.expBase, exp_coeff) + self.meanshape

        face_shape = face_shape.view(n_b, -1, 3)
        face_shape = face_shape - \
            self.meanshape.view(1, -1, 3).mean(dim=1, keepdim=True)

        return face_shape

    def get_color(self, tex_coeff):
        n_b = tex_coeff.size(0)
        face_texture = torch.einsum(
            'ij,aj->ai', self.texBase, tex_coeff) + self.meantex

        face_texture = face_texture.view(n_b, -1, 3)
        return face_texture

    def get_skinmask(self):
        return self.skinmask

    def init_coeff_dims(self):
        self.id_dims = 80
        self.tex_dims = 80
        self.exp_dims = 64
