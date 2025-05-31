%%此文档是选取patch的文档!
clear all;
clc;
f=freadenvi('Washington');
f1=freadenvi('10%train_added_image');
[m,n]=size(f);
[m1,n1]=size(f1);
m01=307;
m02=280;
f11=zeros(m02,m01,n);
h=7;
w=7;
for i=1:n
    f11(:,:,i)=reshape(f(:,i),307,280)';
end
for i=1:n1
    f12(:,:,i)=reshape(f1(:,i),307,280)';
end
f13=zeros(m02+h-1,m01+w-1,n);
f13(floor(h/2)+1:floor(h/2)+m02,floor(w/2)+1:floor(w/2)+m01,:)=f11(:,:,:);
[A1]=find(f12==1);%索引是按列存储的，index=610*(y-1)+x,如16079=（27-1）*610+219（x=219,y=27）
[A2]=find(f12==2);%x的取值范围为1:610，y的取值范围为1:340
[A3]=find(f12==3);
[A4]=find(f12==4);
[A5]=find(f12==5);
[A6]=find(f12==6);
A_added=[A1;A2;A3;A4;A5;A6];
save A_added A_added '-v7.3';
%将testimage转换为与原始影像向对应的610*340
f=freadenvi('Washington');
f2=freadenvi('10%test_added_image');
[m,n]=size(f);
[m2,n2]=size(f2);
m01=307;
m02=280;
for i=1:n2
    f22(:,:,i)=reshape(f2(:,i),307,280)';
end
[B1]=find(f22==1);
[B2]=find(f22==2);
[B3]=find(f22==3);
[B4]=find(f22==4);
[B5]=find(f22==5);
[B6]=find(f22==6);
B_added=[B1;B2;B3;B4;B5;B6];
save B_added B_added '-v7.3';
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
%对镜像之后的影像进行归一化操作
for i=1:m02+h-1
    for j=1:m01+w-1
        f13(i,j,:)=f13(i,j,:)/max(abs((f13(i,j,:))));%对镜像之后的影像进行最大值归一化
    end
end
save inputimage f13 '-v7.3' ;

imagepatch=zeros(w,h,n,m);%从原始影像镜像之后的影像上选取patch
count=0;
for y=1:m01 
    for x=1:m02
        temp=f13(x:x+w-1,y:y+h-1,:);
        %输出一个matlab矩阵查看，固定第一列，然后逐行索引;再进行第二列
        count=count+1;
        imagepatch(:,:,:,count)=temp;
    end
end
save imagepatch imagepatch '-v7.3';
total_num1=25;
trainpatch_added=zeros(w,h,n,total_num1);
for j=1:total_num1
    trainpatch_added(:,:,:,j)=imagepatch(:,:,:,A_added(j));
end
save trainpatch_added trainpatch_added '-v7.3';
trainData_added=reshape(trainpatch_added,h*w*n,total_num1);
save trainData_added trainData_added '-v7.3';
total_num2=32;
testpatch_added=zeros(w,h,n,total_num2);
for j=1:total_num2
    testpatch_added(:,:,:,j)=imagepatch(:,:,:,B_added(j));
end
save testpatch_added testpatch_added '-v7.3';
testData_added=reshape(testpatch_added,h*w*n,total_num2);
save testData_added testData_added '-v7.3';
load trainpatch;
load testpatch;
trainpatch_new(:,:,:,1:3425)=trainpatch;
trainpatch_new(:,:,:,3425+1:3425+25)=trainpatch_added;
testpatch_new(:,:,:,1:30818)=testpatch;
testpatch_new(:,:,:,30818+1:30818+32)=testpatch_added;
save trainpatch_new trainpatch_new '-v7.3';
save testpatch_new testpatch_new '-v7.3';
load trainData;
load testData;
trainData_new=[trainData,trainData_added];
testData_new=[testData,testData_added];
save trainData_new trainData_new '-v7.3';
save testData_new testData_new '-v7.3';


