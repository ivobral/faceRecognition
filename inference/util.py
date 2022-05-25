from PIL import ImageDraw, ImageFont


def draw_bb_on_img(bb, img, prediction, idx_to_class):

    draw = ImageDraw.Draw(img)
    fs = max(20, round(img.size[0] * img.size[1] * 0.000005))
    font = ImageFont.truetype('fonts/font.ttf', fs)
    margin = 5
    top_label = prediction.argmax()

    text = "%s %.2f%%" % (
        idx_to_class[top_label], prediction[top_label] * 100)
    text_size = font.getsize(text)

    # bounding box
    draw.rectangle(
        (
            (int(bb[0]), int(bb[1])),
            (int(bb[2]), int(bb[3]))
        ),
        outline='green',
        width=2
    )

    # text background
    draw.rectangle(
        (
            (int(bb[0] - margin), int(bb[3]) + margin),
            (int(bb[0] + text_size[0] + margin),
                int(bb[3]) + text_size[1] + 3 * margin)
        ),
        fill='black'
    )

    text
    draw.text(
        (int(bb[0]), int(bb[3]) + 2 * margin),
        text,
        font=font
    )
