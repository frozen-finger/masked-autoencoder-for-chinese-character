from PIL import Image
import pygame
import os

dict = {}
for i in range(0x4E00, 0x9FBF+1):
    if chr(i) != "":
        dict[chr(i)] = 1


def paste(text, font, area=(16, 16)):
    im = Image.new("RGB", (224, 224), (255, 255, 255))
    rtext = font.render(text, True, (0, 0, 0), (255, 255, 255))
    path = os.path.join('Fontimage', 'chinese', text+'(1)' + '.png')
    pygame.image.save(rtext, path)
    line = Image.open(path)
    im.paste(line)
    #im.show()
    im.save(path)

def pasteWord(word):
    pygame.init()
    font = pygame.font.SysFont("华文宋体", 168)
    text = word.encode('utf-8').decode('utf-8')
    paste(text, font)

for i in dict:
    if i != ' ':
        pasteWord(i)