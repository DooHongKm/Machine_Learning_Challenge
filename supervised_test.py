import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from torch.utils.data import Dataset, DataLoader

#사용자 정의 데이터셋 클래스
class CustomDataset(Dataset):
    #클래스의 생성자
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = os.listdir(root)
        self.class_to_idx = {c: int(c) for i, c in enumerate(self.classes)}
        self.imgs = []
        for c in self.classes:
            class_dir = os.path.join(root, c)
            for filename in os.listdir(class_dir):
                path = os.path.join(class_dir, filename)
                self.imgs.append((path, self.class_to_idx[c]))
    #데이터셋의 총 샘플 수를 반환
    def __len__(self):
        return len(self.imgs)
    #주어진 index에 해당하는 샘플을 반환
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target

# 사용자 정의 모듈 클래스
class Identity(nn.Module):
    # 클래스의 생성자, nn.Module의 속성과 메서드를 상속
    def __init__(self):
        super(Identity, self).__init__()
    # 모델의 순전파 동작을 정의, 입력을 처리한 결과를 반환
    def forward(self, x):
        return x

#3가지 모델(ResNet, VGG, MobileNet) 중에서 한 개의 모델을 선택하는 함수, 커스텀 모델도 추가 가능
def model_selection(selection):
    if selection == "resnet":
        model = models.resnet18()
        model.conv1 =  nn.Conv2d(3, 64, kernel_size=3,stride=1, padding=1, bias=False)
        model.layer4 = Identity()
        model.fc = nn.Linear(256, 50)
    elif selection == "vgg":
        model = models.vgg11_bn()
        model.features = nn.Sequential(*list(model.features.children())[:-7])
        model.classifier = nn.Sequential( nn.Linear(in_features=25088, out_features=50, bias=True))
    elif selection == "mobilenet":
        model = models.mobilenet_v2()
        model.classifier = nn.Sequential(nn.Linear(in_features=1280, out_features=50, bias=True))
    return model

#딥러닝 모델을 학습하는 과정을 구현한 함수
def train(net1, labeled_loader, optimizer, criterion):
    net1.train()
    for batch_idx, (inputs, targets) in enumerate(labeled_loader):
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()  #그라디언트 초기화
        outputs = net1(inputs)  #입력 데이터를 순전파
        loss = criterion(outputs, targets)  #모델의 예측 결과와 타겟(정답) 사이의 손실을 계산
        loss.backward()  #손실에 대한 그라디언트 계산
        optimizer.step()  #가중치 업데이트
        
#학습된 딥러닝 모델을 평가하는 과정을 구현한 함수
def test(net, testloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        return 100. * correct / total
    
#메인 코드
if __name__ == "__main__":

    #argparse를 사용하여 명령행 인수를 파싱하고, 두 개의 인수를 저장
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default='True')
    parser.add_argument('--student_abs_path', type=str, default='./')
    args = parser.parse_args()

    #로그 디렉토리를 생성하고, 학습 및 평가 과정에서 발생하는 로그를 저장
    if not os.path.exists(os.path.join(args.student_abs_path, 'logs', 'Supervised_Learning')):
        os.makedirs(os.path.join(args.student_abs_path, 'logs', 'Supervised_Learning'))

    #배치 크기를 지정
    #큰 batch_size는 속도는 빠르지만, 정보 손실 가능성 존재
    #작은 batch_size는 속도가 느리지만, 정보 손실 가능성 감소
    batch_size = 10

    #args.test가 "False"인 경우, 모델을 학습하기 위한 데이터 변환 작업
    if args.test == 'False':
        train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(64, scale=(0.2, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        dataset = CustomDataset(root = './data/Supervised_Learning/labeled', transform = train_transform)
        labeled_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        dataset = CustomDataset(root = './data/Supervised_Learning/val', transform = test_transform)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    #args.test가 "True"인 경우, 모델을 테스트하기 위한 데이터 변환 작업
    else :
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    #사용할 모델을 지정, model 종류 : ResNet, VGG, MobileNet, ...
    model_name = "resnet"

    #GPU 사용이 가능한 경우, 모델을 CUDA 장치로 이동
    if torch.cuda.is_available():
        model = model_selection(model_name).cuda()
    else :
        model = model_selection(model_name)

    #모델의 파라미터 수를 계산하여 'params'에 저장
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    
    #GPU 사용이 가능한 경우, 손실함수를 CUDA 장치로 초기화
    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda()
    else :
        criterion = nn.CrossEntropyLoss()

    #학습 횟수인 epoch 값 지정
    epoch = 50

    #optimizer 종류 : SGD, Adam, RMSprop
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #scheduler 종류 : StepLR, ReduceLROnPlateau, CosineAnnealingLR
    scheduler = None  #optimizer가 Adam이기 때문
    
    #최고 성능을 저장하기 위한 변수 초기화
    best_result = 0

    #모델을 학습하기 위한 코드
    if args.test == 'False':

        #모델의 파라미터 수가 7.0M을 초과하면 에러 메세지 출력
        assert params < 7.0, "Exceed the limit on the number of model parameters" 

        #주어진 epoch 횟수만큼 반복
        for e in range(0, epoch):
            
            #train 함수를 호출하여 모델을 학습
            train(model, labeled_loader, optimizer, criterion)
            
            tmp_res = test(model, val_loader) #학습된 모델을 검증 데이터로 평가하여 성능을 확인하고, 정확도를 저장
            print('{}th performance, res : {}'.format(e, tmp_res))  #현재 epoch에서의 성능 결과를 출력

            #이전 최고 성능보다 현재 epoch 성능이 더 좋으면,
            #best_result를 업데이트하고, 모델의 상태를 저장
            if best_result < tmp_res:
                best_result = tmp_res
                torch.save(model.state_dict(),  os.path.join('./logs', 'Supervised_Learning', 'best_model.pt'))

        #학습률 업데이트
        if scheduler is not None: scheduler.step()        
        print('Final performance {} - {}'.format(best_result, params))  #최종 성능 결과와 모델의 파라미터 수를 출력

    #모델을 평가하기 위한 코드
    else:
        dataset = CustomDataset(root = './data/Supervised_Learning/val', transform = test_transform)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        model.load_state_dict(torch.load(os.path.join(args.student_abs_path, 'logs', 'Supervised_Learning', 'best_model.pt'), map_location=torch.device('cuda')))
        res = test(model, test_loader)
        print(res, ' - ' , params)