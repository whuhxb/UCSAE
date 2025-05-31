%%���ĵ���ѡȡpatch���ĵ���
clear all;
clc;
f=freadenvi('Washington');
f1=freadenvi('10%trainimage');
[m,n]=size(f);
[m1,n1]=size(f1);
m01=307;
m02=280;
f11=zeros(m02,m01,n);
%��ԭʼӰ��ת��Ϊ307*280
for i=1:n
    f11(:,:,i)=reshape(f(:,i),307,280)';
end
% enviwrite(f11,610,340,6,'00');%��ʱ��envi����ʾ��ʾ����340�У�610��
% figure;imshow(f11(:,:,1),[]);%matlab����ʾ610��340�У���envi�е�Ӱ�������෴
%��trainimageת��Ϊ��ԭʼӰ�����Ӧ��610*340
for i=1:n1
    f12(:,:,i)=reshape(f1(:,i),307,280)';
end
% enviwrite(f12,610,340,1,'11');%��ʱ��envi����ʾ��ʾ����340�У�610��
% figure;imshow(f12(:,:),[]);%matlab����ʾ610��340�У���envi�е�Ӱ�������෴
% �ȶ�ԭʼӰ���ڱ�Ե��������
h=7;
w=7;
f13=zeros(m02+h-1,m01+w-1,n);
f13(floor(h/2)+1:floor(h/2)+m02,floor(w/2)+1:floor(w/2)+m01,:)=f11(:,:,:);
[A1]=find(f12==1);%�����ǰ��д洢�ģ�index=610*(y-1)+x,��16079=��27-1��*610+219��x=219,y=27��
[A2]=find(f12==2);%x��ȡֵ��ΧΪ1:610��y��ȡֵ��ΧΪ1:340
[A3]=find(f12==3);
[A4]=find(f12==4);
[A5]=find(f12==5);
[A6]=find(f12==6);
A=[A1;A2;A3;A4;A5;A6];
save A A '-v7.3';
%�õ���������������
%groundtruth�еĲ���������patches
f=freadenvi('Washington');
f2=freadenvi('10%testimage');
[m,n]=size(f);
[m2,n2]=size(f2);
m01=307;
m02=280;
%��testimageת��Ϊ��ԭʼӰ�����Ӧ��610*340
for i=1:n2
    f22(:,:,i)=reshape(f2(:,i),307,280)';
end
[B1]=find(f22==1);
[B2]=find(f22==2);
[B3]=find(f22==3);
[B4]=find(f22==4);
[B5]=find(f22==5);
[B6]=find(f22==6);
B=[B1;B2;B3;B4;B5;B6];
save B B '-v7.3';
f=freadenvi('Washington');
fg=freadenvi('gtimage');
[m,n]=size(f);
[mg,ng]=size(fg);
m01=307;
m02=280;
%��testimageת��Ϊ��ԭʼӰ�����Ӧ��610*340
for i=1:ng
    fgg(:,:,i)=reshape(fg(:,i),307,280)';
end
[G1]=find(fgg==1);
[G2]=find(fgg==2);
[G3]=find(fgg==3);
[G4]=find(fgg==4);
[G5]=find(fgg==5);
[G6]=find(fgg==6);
G=[G1;G2;G3;G4;G5;G6];
save G G '-v7.3';
%copy the rows
for i=1:n
    f13(1:floor(h/2),floor(w/2)+1:floor(w/2)+m01,i)=flipud(f11(1:floor(h/2),:,i));
    f13(floor(h/2)+m02+1:m02+h-1,floor(w/2)+1:floor(w/2)+m01,i)=flipud(f11(m02-floor(h/2)+1:m02,:,i)); 
end
%copy the columns
for i=1:n
    f13(:,1:floor(w/2),i)=fliplr(f13(:,floor(w/2)+1:w-1,i));
    f13(:,m01+floor(w/2)+1:m01+w-1,i)=fliplr(f13(:,m01+1:floor(w/2)+m01,i));
end
%�Ծ���֮���Ӱ����й�һ������
for i=1:m02+h-1
    for j=1:m01+w-1
        f13(i,j,:)=f13(i,j,:)/max(abs((f13(i,j,:))));%�Ծ���֮���Ӱ��������ֵ��һ��
    end
end
save inputimage f13 '-v7.3';

imagepatch=zeros(w,h,n,m);%��ԭʼӰ����֮���Ӱ����ѡȡpatch
count=0;
for y=1:m01 
    for x=1:m02
        temp=f13(x:x+w-1,y:y+h-1,:);
        %���һ��matlab����鿴���̶���һ�У�Ȼ����������;�ٽ��еڶ���
        count=count+1;
        imagepatch(:,:,:,count)=temp;
    end
end
save imagepatch imagepatch '-v7.3';
total_num1=3425;
trainpatch=zeros(w,h,n,total_num1);
for i=1:total_num1
    trainpatch(:,:,:,i)=imagepatch(:,:,:,A(i));
end
save trainpatch trainpatch '-v7.3';
trainData=reshape(trainpatch,h*w*n,total_num1);
save trainData trainData '-v7.3';
total_num2=30818;
testpatch=zeros(w,h,n,total_num2);
for j=1:total_num2
    testpatch(:,:,:,j)=imagepatch(:,:,:,B(j));
end
save testpatch testpatch '-v7.3';
testData=reshape(testpatch,h*w*n,total_num2);
save testData testData '-v7.3';
inputpatch1=[trainData,testData];
inputpatch1=reshape(inputpatch1,h,w,n,total_num1+total_num2);
save inputpatch1 inputpatch1 '-v7.3';

total_numg=3425+30818;
gtpatch=zeros(w,h,n,total_numg);
for j=1:total_numg
     gtpatch(:,:,:,j)=imagepatch(:,:,:,G(j));
end
save gtpatch gtpatch '-v7.3';
% clear all;