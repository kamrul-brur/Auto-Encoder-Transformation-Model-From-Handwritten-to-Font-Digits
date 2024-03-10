from PIL import Image, ImageDraw, ImageFont


def create_image_with_digits(digit, image_size):
    """
    Create an image with a specified digit and save it to a directory.

    Parameters:
    - digit (int): The digit to be placed in the center of the image. Should be in the range [0, 25] which denotes [A...Z].
    - image_size (tuple): A tuple representing the size of the image (width, height).

    Saves the image to the './synthetic_digits/' directory with a filename corresponding to the digit.

    Example:
    # >>> create_image_with_letters(0, (28, 28))
    # Creates an image with the letter 'A' and saves it as '1.jpg' in the './synthetic_digits/' directory.
    """
    image = Image.new('L', image_size, color='black')
    draw = ImageDraw.Draw(image)
    text = str(chr(digit + 48))
    font_size = 30  # Adjust the font size here
    font = ImageFont.truetype('./Times_New_Roman.ttf', font_size)

    text_width = int(draw.textlength(text, font=font))
    text_height = font_size
    text_x = (image_size[0] - text_width) // 2
    text_y = (image_size[1] - text_height) // 2 - 2

    draw.text((text_x, text_y), text, fill='white', font=font, antialias=False)
    image.save('./synthetic_digits/' + f'{digit}.jpg')


for label in range(0, 10):
    create_image_with_digits(label, (28, 28))
print('Image Generated Successfully!!!')
