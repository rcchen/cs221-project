jpegFiles = dir('photos/*.jpg');
numFiles = length(jpegFiles);

for i = 1:numFiles
    seg(jpegFiles(i).name);
end