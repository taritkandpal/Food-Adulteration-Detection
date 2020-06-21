x=imageSet('Food\Dataset\train\freshapples','recursive');
y=imageSet('Food\Dataset\train\rottenapples','recursive');
j=read(x(1),1);
i=rgb2gray(j);
e=entropy(i);
m=mean2(i);
s=std2(i);
A=[e m s 1];
for c=2:1693
    j=read(x(1),c);
    i=rgb2gray(j);
    e=entropy(i);
    m=mean2(i);
    s=std2(i);
    B=[e m s 1];
    A=[A ; B];
end
j=read(y(1),1);
i=rgb2gray(j);
e=entropy(i);
m=mean2(i);
s=std2(i);
D=[e m s 0];
for c=2:2342
    j=read(y(1),c);
    i=rgb2gray(j);
    e=entropy(i);
    m=mean2(i);
    s=std2(i);
    E=[e m s 0];
    D=[D ; E];
end
F=[A ; D];
export = F(randperm(size(F,1)),:);
csvwrite('apple.csv',export);