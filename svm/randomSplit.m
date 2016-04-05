function [training,testing] = randomSplit(data,portion)
    training = [];
    testing = [];
    datasize = size(data);
    numofrow = datasize(1);
    numofcol = datasize(2);
    for i=1:1:numofrow
        if(rand()<portion)
        	training = [training;data(i,:)];
        else
            testing = [testing;data(i,:)];
        end
end