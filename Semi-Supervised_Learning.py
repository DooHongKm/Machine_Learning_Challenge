import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from torch.utils.data import Dataset, DataLoader, ConcatDataset

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
    
#레이블이 없는 데이터셋을 처리하기 위한 사용자 정의 데이터셋 클래스
class CustomDataset_Nolabel(Dataset):
    #클래스의 생성자
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        ImageList = os.listdir(root)
        self.imgs = []
        for filename in ImageList:
            path = os.path.join(root, filename)
            self.imgs.append(path)
    #데이터셋의 총 샘플 수를 반환
    def __len__(self):
        return len(self.imgs)
    #주어진 index에 해당하는 샘플을 반환    
    def __getitem__(self, index):
        path = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

#사용자 정의 모듈 클래스
class Identity(nn.Module):
    #클래스의 생성자, nn.Module의 속성과 메서드를 상속
    def __init__(self):
        super(Identity, self).__init__()
    #모델의 순전파 동작을 정의, 입력을 처리한 결과를 반환
    def forward(self, x):
        return x

#3가지 모델(ResNet, VGG, MobileNet) 중에서 한 개의 모델을 선택하는 함수, 커스텀 모델도 추가 가능
def model_selection(selection):
    if selection == "resnet":
        model = models.resnet18()
        model.conv1 =  nn.Conv2d(3, 64, kernel_size=3,stride=1, padding=1, bias=False)
        model.layer4 = Identity()
        model.fc = nn.Linear(256, 10)
    elif selection == "vgg":
        model = models.vgg11_bn()
        model.features = nn.Sequential(*list(model.features.children())[:-7])
        model.classifier = nn.Sequential( nn.Linear(in_features=25088, out_features=10, bias=True))
    elif selection == "mobilenet":
        model = models.mobilenet_v2()
        model.classifier = nn.Sequential( nn.Linear(in_features=1280, out_features=10, bias=True))
    return model

# 딥러닝 모델을 협력학습하는 과정을 구현한 함수
def cotrain(net1,net2, labeled_loader, unlabeled_loader, optimizer1, optimizer2, criterion):

    #labeled_training
    net1.train()
    net2.train()
    for batch_idx, (inputs, targets) in enumerate(labeled_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer1.zero_grad()  #그라디언트 초기화
        optimizer2.zero_grad()
        outputs1 = net1(inputs)  #입력 데이터를 순전파
        outputs2 = net2(inputs)
        loss1 = criterion(outputs1, targets)  #모델의 예측 결과와 타겟(정답) 사이의 손실을 계산
        loss2 = criterion(outputs2, targets)
        loss1.backward()  #손실에 대한 그라디언트 계산
        loss2.backward()
        optimizer1.step()  #가중치 업데이트
        optimizer2.step()
    
    #unlabeled_testing    
    net1.eval()
    net2.eval()
    pseudo_labeled_dataset = []
    with torch.no_grad():
        for batch_idx, (inputs) in enumerate(unlabeled_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            _, predicted1 = outputs1.max(1)
            _, predicted2 = outputs2.max(1)
            mask = (predicted1 == targets) & (predicted2 == targets)
            count = 0
            for input, target, m in zip(inputs, targets, mask):
                if m:
                    pseudo_labeled_dataset.extend([(input.cpu(), target.cpu().item())])
                    count += 1
    
    #pseudo training
    if count > 0:
        pseudo_labeled_loader = DataLoader(pseudo_labeled_dataset, batch_size=count, shuffle=True, num_workers=4, pin_memory=True)
        net1.train()
        net2.train()                
        for batch_idx, (inputs, targets) in enumerate(pseudo_labeled_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer1.zero_grad()  #그라디언트 초기화
            optimizer2.zero_grad()
            outputs1 = net1(inputs)  #입력 데이터를 순전파
            outputs2 = net2(inputs)
            loss1 = criterion(outputs1, targets)  #모델의 예측 결과와 타겟(정답) 사이의 손실을 계산
            loss2 = criterion(outputs2, targets)
            loss1.backward()  #손실에 대한 그라디언트 계산
            loss2.backward()
            optimizer1.step()  #가중치 업데이트
            optimizer2.step()
    
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
    parser.add_argument('--test',  type=str,  default='False')
    parser.add_argument('--student_abs_path',  type=str,  default='./')
    args = parser.parse_args()   

    #로그 디렉토리를 생성하고, 학습 및 평가 과정에서 발생하는 로그를 저장
    if not os.path.exists(os.path.join(args.student_abs_path, 'logs', 'Semi-Supervised_Learning')):
        os.makedirs(os.path.join(args.student_abs_path, 'logs', 'Semi-Supervised_Learning'))

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
        dataset = CustomDataset(root = './data/Semi-Supervised_Learning/labeled', transform = train_transform)
        labeled_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        dataset = CustomDataset_Nolabel(root = './data/Semi-Supervised_Learning/unlabeled', transform = train_transform)
        unlabeled_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)        
        dataset = CustomDataset(root = './data/Semi-Supervised_Learning/val', transform = test_transform)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    #args.test가 "True"인 경우, 모델을 테스트하기 위한 데이터 변환 작업
    else :
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])    
    
    #사용할 모델을 2개 지정, model 종류 : ResNet, VGG, MobileNet
    model_sel_1 =  "resnet"
    model_sel_2 =  "resnet"
    model1 = model_selection(model_sel_1)
    model2 = model_selection(model_sel_2)
    
    #모델의 파라미터 수를 계산하여 'params'에 저장
    params_1 = sum(p.numel() for p in model1.parameters() if p.requires_grad) / 1e6
    params_2 = sum(p.numel() for p in model2.parameters() if p.requires_grad) / 1e6

    #GPU 사용이 가능한 경우, 모델을 CUDA 장치로 이동 
    if torch.cuda.is_available():
        model1 = model1.cuda()
    if torch.cuda.is_available():
        model2 = model2.cuda()
 
    #GPU 사용이 가능한 경우, 손실함수를 CUDA 장치로 초기화
    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda()
    else :
        criterion = nn.CrossEntropyLoss()    
        
    #학습 횟수인 epoch 값 지정
    epoch = 50
    
    #optimizer 종류 : SGD, Adam, RMSprop
    optimizer1 = optim.RMSprop(model1.parameters(), lr=0.001)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

    #scheduler 종류 : StepLR, ReduceLROnPlateau, CosineAnnealingLR
    scheduler1 = None  #optimizer1이 RMSprop이기 때문
    scheduler2 = None  #optimizer2가 Adam이기 때문

    #최고 성능을 저장하기 위한 변수 초기화
    best_result_1 = 0
    best_result_2 = 0
    
    #모델을 학습하기 위한 코드
    if args.test == 'False':

        #모델의 파라미터 수가 7.0M을 초과하면 에러 메세지 출력
        assert params_1 < 7.0, "Exceed the limit on the number of model_1 parameters" 
        assert params_2 < 7.0, "Exceed the limit on the number of model_2 parameters" 
        
        # 주어진 epoch 횟수만큼 반복
        for e in range(0, epoch):
            
            #cotrain 함수를 호출하여 모델을 학습
            cotrain(model1, model2, labeled_loader, unlabeled_loader, optimizer1, optimizer2, criterion)

            tmp_res_1 = test(model1, val_loader)  #학습된 모델1을 검증 데이터로 평가하여 성능을 확인하고, 정확도를 저장
            print ("[{}th epoch, model_1] ACC : {}".format(e, tmp_res_1))  #현재 epoch에서의 성능 결과를 출력

            #이전 최고 성능보다 현재 epoch 성능이 더 좋으면,
            #best_result1을 업데이트하고, 모델1의 상태를 저장
            if best_result_1 < tmp_res_1:
                best_result_1 = tmp_res_1
                torch.save(model1.state_dict(),  os.path.join('./logs', 'Semi-Supervised_Learning', 'best_model_1.pt'))

            tmp_res_2 = test(model2, val_loader)  #학습된 모델2를 검증 데이터로 평가하여 성능을 확인하고, 정확도를 저장
            print ("[{}th epoch, model_2] ACC : {}".format(e, tmp_res_2))  #현재 epoch에서의 성능 결과를 출력

            #이전 최고 성능보다 현재 epoch 성능이 더 좋으면,
            #best_result2를 업데이트하고, 모델2의 상태를 저장           
            if best_result_2 < tmp_res_2:
                best_result_2 = tmp_res_2
                torch.save(model2.state_dict(),  os.path.join('./logs', 'Semi-Supervised_Learning', 'best_model_2.pt'))

            #학습률 업데이트
            if scheduler1 is not None: scheduler1.step()
            if scheduler2 is not None: scheduler2.step()
            
        print(f'Final performance {best_result_1} - {params_1}  // {best_result_2} - {params_2}')  # 최종 성능 결과와 모델의 파라미터 수를 출력

    # 모델을 평가하기 위한 코드        
    else:
        dataset = CustomDataset(root = '/data/23_1_ML_challenge/Semi-Supervised_Learning/test', transform = test_transform)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        model1.load_state_dict(torch.load(os.path.join(args.student_abs_path, 'logs', 'Semi-Supervised_Learning', 'best_model_1.pt'), map_location=torch.device('cuda')))
        res1 = test(model1, test_loader)
        model2.load_state_dict(torch.load(os.path.join(args.student_abs_path, 'logs', 'Semi-Supervised_Learning', 'best_model_2.pt'), map_location=torch.device('cuda')))
        res2 = test(model2, test_loader)
        if res1>res2:
            best_res = res1
            best_params = params_1
        else :
            best_res = res2
            best_params = params_2
        print(best_res, ' - ', best_params)