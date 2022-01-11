import ffmpeg
import subprocess
from inpainter.options import Options

opt = Options()


folder = f'{opt.ouput}'
src_video = 'media/src_english_female_cut_0106.mp4'
#src_video = 'media/src_chinese_female.mp4'


subprocess.call(
    f'ffmpeg -i {folder}/videocompare.mp4 -i {src_video} -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 {folder}/deflicker_output_compare.mp4',
    shell=True)