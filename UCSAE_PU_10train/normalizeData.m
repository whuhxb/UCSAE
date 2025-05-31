function patches = normalizeData(patches)%���ֹ�һ��Ϊȫ�ֵĹ�һ����ƽ��ֵ���ǰ�����ģ����Ǳ�׼�����ȫ�ֵı�׼��
patches = bsxfun(@minus, patches, mean(patches));
% Truncate to +/-3 standard deviations and scale to -1 to 1
pstd = 3 * std(patches(:));%patches(:)��ʾ�Ѿ���չ����һ�У�������reshape��Ȼ������std�����������׼�
patches = max(min(patches, pstd), -pstd) / pstd;%������һ����patches���Ϊpstd����СΪ-pstd���ٳ���pstd���õ����Ϊ1����СΪ-1.
% Rescale from [-1,1] to [0.1,0.9]
patches = (patches + 1) * 0.4 + 0.1;
end