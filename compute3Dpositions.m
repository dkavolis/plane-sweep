function [x,y,z] = compute3Dpositions(txt_file,depth_file)

    %depth_file = 'scene_000.depth';

    K = getcamK(txt_file);
    
    
    fx = K(1,1);
    fy = K(2,2);
    u0 = K(1,3);
    v0 = K(2,3);
    
    
    u = repmat([1:640],480,1);
    v = repmat([1:480]',1,640);
    
    u_u0_by_fx = (u - u0)/fx;
    v_v0_by_fy = (v - v0)/fy;
    
    size(u);
    size(v);
    
%     depth_file_endianess = 'b' ;
%     f = fopen( depth_file ) ;
%     z = fread( f, 'double', depth_file_endianess ) ;
%     fclose(f) ;

    z = load(depth_file);
    
    z = reshape(z,640,480)' ;  %z is radial 
        
    z = z ./ sqrt(u_u0_by_fx.^2 + v_v0_by_fy.^2 + 1);
    
    x = ((u-u0)/fx).*z;
    y = ((v-v0)/fy).*z;


end