import ffmpeg
import subprocess
from inpainter.options import Options

opt = Options()


folder = f'{opt.ouput}'
src_video = 'media/src_trump_chinse_v2.mp4'


subprocess.call(
    f'ffmpeg -i {folder}/videorendered.mp4 -i {src_video} -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 {folder}/output.mp4',
    shell=True)