function result = flatten(x,len,N)
    result = reshape(squeeze(x(:,:,1,:)),len,N);
end