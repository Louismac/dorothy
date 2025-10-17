class DOTSHADERS:

    VERT_3D = '''
                #version 330
                
                uniform mat4 model;
                uniform mat4 view;
                uniform mat4 projection;
                
                in vec3 in_position;
                in vec3 in_normal;
                
                out vec3 v_normal;
                out vec3 v_position;
                
                void main() {
                    vec4 world_pos = model * vec4(in_position, 1.0);
                    v_position = world_pos.xyz;
                    v_normal = mat3(transpose(inverse(model))) * in_normal;
                    gl_Position = projection * view * world_pos;
                }
            '''

    FRAG_3D = '''
                #version 330
                
                uniform vec4 color;
                uniform vec3 light_pos;
                uniform vec3 camera_pos;
                uniform bool use_lighting;
                
                in vec3 v_normal;
                in vec3 v_position;
                
                out vec4 fragColor;
                
                void main() {
                    if (use_lighting) {
                        vec3 normal = normalize(v_normal);
                        vec3 light_dir = normalize(light_pos - v_position);
                        vec3 view_dir = normalize(camera_pos - v_position);
                        vec3 reflect_dir = reflect(-light_dir, normal);
                        
                        // Ambient
                        float ambient = 0.3;
                        
                        // Diffuse
                        float diffuse = max(dot(normal, light_dir), 0.0);
                        
                        // Specular
                        float specular = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0) * 0.5;
                        
                        float lighting = ambient + diffuse + specular;
                        fragColor = vec4(color.rgb * lighting, color.a);
                    } else {
                        fragColor = color;
                    }
                }
            '''
    
    VERT_2D = '''
                #version 330
                
                uniform mat4 projection;
                uniform mat4 model;
                
                in vec2 in_position;
                
                void main() {
                    gl_Position = projection * model * vec4(in_position, 0.0, 1.0);
                }
            '''
    
    FRAG_2D = '''
                #version 330
                
                uniform vec4 color;
                out vec4 fragColor;
                
                void main() {
                    fragColor = color;
                }
            '''
    
    VERT_TEXTURE = '''
                #version 330
                
                in vec2 in_position;
                in vec2 in_texcoord;
                
                out vec2 v_texcoord;
                
                void main() {
                    v_texcoord = in_texcoord;
                    gl_Position = vec4(in_position, 0.0, 1.0);
                }
            '''
    
    FRAG_TEXTURE = '''
                #version 330
                
                uniform sampler2D texture0;
                uniform float alpha;
                
                in vec2 v_texcoord;
                out vec4 fragColor;
                
                void main() {
                    vec4 texColor = texture(texture0, v_texcoord);
                    // Discard fully transparent pixels so they don't overwrite the screen
                    if (texColor.a < 0.01) {
                        discard;
                    }
                    fragColor = vec4(texColor.rgb, texColor.a * alpha);
                }
            '''
    
    VERT_TEXTURE_TRANSFORM = '''
                #version 330
                
                uniform mat4 projection;
                uniform mat4 model;
                
                in vec2 in_position;
                in vec2 in_texcoord;
                
                out vec2 v_texcoord;
                
                void main() {
                    v_texcoord = in_texcoord;
                    gl_Position = projection * model * vec4(in_position, 0.0, 1.0);
                }
            '''
    
    FRAG_TEXTURE_TRANSFORM = '''
                #version 330
                
                uniform sampler2D texture0;
                uniform float alpha;
                
                in vec2 v_texcoord;
                out vec4 fragColor;
                
                void main() {
                    vec4 texColor = texture(texture0, v_texcoord);
                    // Discard fully transparent pixels so they don't overwrite the screen
                    if (texColor.a < 0.01) {
                        discard;
                    }
                    fragColor = vec4(texColor.rgb, texColor.a * alpha);
                }
            '''
    
    VERT_TEXTURE_2D = '''
                    #version 330
                    
                    uniform mat4 projection;
                    uniform mat4 model;
                    
                    in vec2 in_position;
                    in vec2 in_texcoord;
                    
                    out vec2 v_texcoord;
                    
                    void main() {
                        v_texcoord = in_texcoord;
                        gl_Position = projection * model * vec4(in_position, 0.0, 1.0);
                    }
                '''
    
    FRAG_TEXTURE_2D = '''
                    #version 330
                    
                    uniform sampler2D texture0;
                    uniform float alpha;
                    
                    in vec2 v_texcoord;
                    out vec4 fragColor;
                    
                    void main() {
                        vec4 texColor = texture(texture0, v_texcoord);
                        fragColor = vec4(texColor.rgb, texColor.a * alpha);
                    }
                '''
    
    PIXELATE = '''
    #version 330
    uniform sampler2D texture0;
    uniform vec2 resolution;
    uniform float pixelSize;
    in vec2 v_texcoord;
    out vec4 fragColor;

    void main() {
        vec2 pixels = resolution / pixelSize;
        vec2 uv = floor(v_texcoord * pixels) / pixels;
        fragColor = texture(texture0, uv);
    }
    '''

    BLUR = '''
    #version 330
    uniform sampler2D texture0;
    uniform vec2 resolution;
    in vec2 v_texcoord;
    out vec4 fragColor;

    void main() {
        vec2 pixel = 1.0 / resolution;
        vec4 color = vec4(0.0);
        
        for(int x = -2; x <= 2; x++) {
            for(int y = -2; y <= 2; y++) {
                color += texture(texture0, v_texcoord + vec2(x, y) * pixel);
            }
        }
        
        fragColor = color / 25.0;
    }
    '''

    RGB_SPLIT = '''
    #version 330
    uniform sampler2D texture0;
    uniform float offset;
    in vec2 v_texcoord;
    out vec4 fragColor;

    void main() {
        float r = texture(texture0, v_texcoord + vec2(offset, 0.0)).r;
        float g = texture(texture0, v_texcoord).g;
        float b = texture(texture0, v_texcoord - vec2(offset, 0.0)).b;
        fragColor = vec4(r, g, b, 1.0);
    }
    '''

    FEEDBACK = '''
        #version 330
        uniform sampler2D texture0;
        uniform float zoom;
        in vec2 v_texcoord;
        out vec4 fragColor;
        
        void main() {
            vec2 uv = (v_texcoord - 0.5) * zoom + 0.5;
            vec4 color = texture(texture0, uv);
            color.rgb *= 0.98;
            fragColor = color;
        }
        '''
    
    INVERT = '''
        #version 330
        uniform sampler2D texture0;
        in vec2 v_texcoord;
        out vec4 fragColor;

        void main() {
            vec4 color = texture(texture0, v_texcoord);
            color.rgb = 1.0 - color.rgb;
            fragColor = color;
        }
        '''
    
    ROLL = '''
    #version 330
    uniform sampler2D texture0;
    uniform vec2 offset;
    uniform vec2 resolution;
    in vec2 v_texcoord;
    out vec4 fragColor;

    void main() {
        // Convert offset from pixels to texture coordinates (0-1)
        vec2 uv_offset = offset / resolution;
        
        // Add offset and wrap with fract (like modulo for 0-1 range)
        vec2 rolled_uv = fract(v_texcoord + uv_offset);
        
        fragColor = texture(texture0, rolled_uv);
    }
    '''

    TILE = '''
    #version 330
    uniform sampler2D texture0;
    uniform vec2 resolution;
    uniform vec2 grid_size;
    in vec2 v_texcoord;
    out vec4 fragColor;

    void main() {
        // Scale UV coordinates by grid size
        vec2 tiled_uv = v_texcoord * grid_size;
        
        // Use fract to repeat the texture in each tile
        vec2 repeated_uv = fract(tiled_uv);
        
        fragColor = texture(texture0, repeated_uv);
    }
    '''

    CUTOUT = '''
    #version 330
    uniform sampler2D texture0;
    uniform vec3 cutout_color;
    uniform float threshold;
    in vec2 v_texcoord;
    out vec4 fragColor;

    void main() {
        vec4 color = texture(texture0, v_texcoord);
        
        // Calculate distance between pixel color and cutout color
        vec3 diff = abs(color.rgb - cutout_color);
        float distance = length(diff);
        
        // If pixel is close to cutout color, make it transparent
        if (distance < threshold) {
            color.a = 0.0;
        }
        
        fragColor = color;
    }
    '''