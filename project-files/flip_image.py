from PIL import Image
def flip_image(image_path, saved_location):
    image_obj = Image.open(image_path)
    rotated_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
    rotated_image.save(saved_location)
    rotated_image.show()


if __name__ == '__main__':
    image = 'left_!/data/0left_1.png'
    flip_image(image, 'left_!/0left_1_flip.png')