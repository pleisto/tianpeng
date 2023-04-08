import base64
from PIL import Image
from io import BytesIO
import json


def get_imagestr_from_sd_resp_file(sd_resp_file) -> str:
    """
    从sd返回的json文件中获取图片的base64编码
    :param sd_resp_file: sd返回的json文件
    :return: 图片的base64编码
    """
    with open(sd_resp_file, "r") as f:
        sd_resp = json.load(f)
    return sd_resp["images"][0]


def get_imagestr_from_sd_resp(sd_resp):
    """
    从sd返回的json中获取图片的base64编码
    :param sd_resp: sd返回的json
    :return: 图片的base64编码
    """
    sd_resp_json = json.loads(sd_resp)
    return sd_resp_json["images"][0]
    # return sd_resp["images"][0]


def get_imagestr_from_http_resp(http_resp):
    get_imagestr_from_sd_resp(http_resp.read().decode("utf-8"))


def turn_base64_to_png(b64s):
    """
    将base64编码的图片转换为png格式
    :param b64s: base64编码的图片
    :param output_path: 输出路径
    :return:
    """
    img = Image.open(BytesIO(base64.b64decode(b64s)))
    return img


def turn_base64_to_png_and_save(b64s, output_path):
    """
    将base64编码的图片转换为png格式
    :param b64s: base64编码的图片
    :param output_path: 输出路径
    :return:
    """
    img = Image.open(BytesIO(base64.b64decode(b64s)))
    img.save(output_path)
