from torchvision import models, transforms
from PIL import Image
from captum.attr import IntegratedGradients

def main():
    model = models.efficientnet_b7(pretrained=True)
    model.eval()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image = Image.open('cat.jpg')
    image = transform(image)
    image = image.unsqueeze(0)
    attr = IntegratedGradients(model).attribute(image)

    #print(output)
    #ans = output.squeeze().detach().numpy().tolist()
    #print(ans.index(max(ans)))
    to_img = transforms.ToPILImage()
    to_img(attr).save('C:/Users/User/Downloads/car_attr.jpg')



if __name__ == '__main__':
    main()