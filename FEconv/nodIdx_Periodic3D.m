function nodIdx = nodIdx_Periodic3D()
num=40;
nod_ele = gen_vox_info_periodicMesh3D(num)+1;
% nod_ele(nod_ele==0) = (num+0)^3+1;
[CUBE,V]=MeshGenerate(num);
% [eleidx,mesh,VE]=periodicMesh(num);
% CUBE=mesh;
% V=VE;
% CUBE(num^3+1,:) = 0;
nodIdx = zeros((num+1)^3,27);

IndexMapping = cell(27,1);
IndexMapping{1} = [1,1];% 1号单元的1号点
IndexMapping{2} = [1,2;
                   2,1];
IndexMapping{3} = [2,2];
IndexMapping{4} = [1,3;
                   3,1];
IndexMapping{5} = [1,4;
                   2,3;
                   3,2;
                   4,1];
IndexMapping{6} = [2,4;
                   4,2];
IndexMapping{7} = [3,3];
IndexMapping{8} = [3,4;
                   4,3];
IndexMapping{9} = [4,4];

IndexMapping{10} = [1,5;
                   5,1];
IndexMapping{11} = [1,6;
                    2,5;
                    5,2;
                   6,1];
IndexMapping{12} = [2,6;
                   6,2];
IndexMapping{13} = [1,7;
                   3,5;
                   5,3;
                   7,1];
IndexMapping{14} = [1,8;
                   2,7;
                   3,6;
                   4,5;
                   5,4;
                   6,3;
                   7,2;
                   8,1];
IndexMapping{15} = [2,8;
                   4,6;
                   6,4;
                   8,2];
IndexMapping{16} = [3,7;
                    7,3];
IndexMapping{17} = [3,8;
                    4,7;
                    7,4;
                   8,3];
IndexMapping{18} = [4,8;
                   8,4];
               
IndexMapping{19} = [5,5];
IndexMapping{20} = [5,6;
                   6,5];
IndexMapping{21} = [6,6];
IndexMapping{22} = [5,7;
                   7,5];
IndexMapping{23} = [5,8;
                   6,7;
                   7,6;
                   8,5];
IndexMapping{24} = [6,8;
                   8,6];
IndexMapping{25} = [7,7];
IndexMapping{26} = [7,8;
                    8,7];
IndexMapping{27} = [8,8];
               
% indexmap_ele = [1,1,2,1,1,2,3,3,4, 1,1,2,1,1,2,3,3,4, 5,5,6,5,5,6,7,7,8];
% indexmap_nod = [1,2,2,3,4,4,3,4,4, 5,6,6,7,8,8,7,8,8, 5,6,6,7,8,8,7,8,8];

% nod_ele(1,:)
for i =1:(num+1)^3
    for j =1:27
%         if i==1 && j==4
%             indexmap_ele(j)
%             nod_ele(i,indexmap_ele(j))
%             indexmap_nod(j)
%             CUBE(nod_ele(i,indexmap_ele(j)),indexmap_nod(j))
%         end
        im = IndexMapping{j};
        for k=1:size(im,1)
            iele = nod_ele(i,im(k,1));
            if  iele ~= 0
%                 disp([i,j,k,im(k,1),im(k,2),iele,CUBE(iele,im(k,2))])
                nodIdx(i,j) = CUBE(iele,im(k,2));
            end
        end   
%         nodIdx(i,j) = CUBE(nod_ele(i,indexmap_ele(j)),indexmap_nod(j));
    end
end
% nodIdx(nodIdx==0) = (num+0)^3+1;
nodIdx(nodIdx==0) = (num+1)^3+1;
nodIdx = nodIdx-1;
save('nodIdx_matrix7_notPeriodic.mat','nodIdx','-v7.3');
end