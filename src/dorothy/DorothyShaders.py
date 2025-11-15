class DOTSHADERS:

    # In DorothyRenderer.__init__, add this shader:

    VERT_3D_INSTANCED = '''
        #version 330
        
        uniform mat4 projection;
        uniform mat4 view;
        
        in vec3 in_position;
        in vec3 in_normal;
        in vec2 in_texcoord_0;
        
        // Per-instance attributes
        in mat4 instance_model;
        in vec4 instance_color;
        
        out vec3 v_position;
        out vec3 v_normal;
        out vec2 v_texcoord;
        out vec4 v_color;
        
        void main() {
            vec4 world_pos = instance_model * vec4(in_position, 1.0);
            v_position = world_pos.xyz;
            v_normal = mat3(instance_model) * in_normal;
            v_texcoord = in_texcoord_0;
            v_color = instance_color;
            
            gl_Position = projection * view * world_pos;
        }
    ''' 

    FRAG_3D_INSTANCED = '''
        #version 330
        uniform sampler2D texture0;
        uniform bool use_texture;
        uniform vec4 color;
        uniform vec3 light_pos;
        uniform vec3 camera_pos;
        uniform bool use_lighting;
        uniform float ambient = 0.3;
        
        in vec3 v_normal;
        in vec3 v_position;
        in vec2 v_texcoord;
        in vec4 v_color;
        
        out vec4 fragColor;
        
        void main() {
            vec4 base_color;
            if (use_texture) {
                base_color = texture(texture0, v_texcoord);
            } else {
                base_color = v_color;  // Use per-instance colour instead of uniform
            }
            
            if (use_lighting) {
                vec3 normal = normalize(v_normal);
                vec3 light_dir = normalize(light_pos - v_position);
                vec3 view_dir = normalize(camera_pos - v_position);
                vec3 reflect_dir = reflect(-light_dir, normal);
                
                // Diffuse
                float diffuse = max(dot(normal, light_dir), 0.0);
                
                // Specular
                float specular = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0) * 0.5;
                
                float lighting = ambient + diffuse + specular;
                fragColor = vec4(base_color.rgb * lighting, base_color.a);
            } else {
                fragColor = base_color;
            }
        }
    '''

    VERT_3D_TEXTURED = '''
        #version 330
        
        uniform mat4 projection;
        uniform mat4 view;
        uniform mat4 model;
        
        in vec3 in_position;
        in vec3 in_normal;
        in vec2 in_texcoord_0;
        
        out vec3 v_position;
        out vec3 v_normal;
        out vec2 v_texcoord;
        
        void main() {
            vec4 world_pos = model * vec4(in_position, 1.0);
            v_position = world_pos.xyz;
            v_normal = mat3(model) * in_normal;
            v_texcoord = in_texcoord_0;
            
            gl_Position = projection * view * world_pos;
        }
    ''' 
    
    FRAG_3D_TEXTURED = '''

                #version 330
                uniform sampler2D texture0;
                uniform bool use_texture;
                uniform vec4 color;
                uniform vec3 light_pos;
                uniform vec3 camera_pos;
                uniform bool use_lighting;
                uniform float ambient=0.3;
                
                in vec3 v_normal;
                in vec3 v_position;
                in vec2 v_texcoord;
                
                out vec4 fragColor;
                
                void main() {
                        vec4 base_color;
                    if (use_texture) {
                        base_color = texture(texture0, v_texcoord);
                    } else {
                        base_color = color;
                    }
                    if (use_lighting) {
                        vec3 normal = normalize(v_normal);
                        vec3 light_dir = normalize(light_pos - v_position);
                        vec3 view_dir = normalize(camera_pos - v_position);
                        vec3 reflect_dir = reflect(-light_dir, normal);
                        
                        // Diffuse
                        float diffuse = max(dot(normal, light_dir), 0.0);
                        
                        // Specular
                        float specular = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0) * 0.5;
                        
                        float lighting = ambient + diffuse + specular;
                        fragColor = vec4(base_color.rgb * lighting, base_color.a);
                    } else {
                        fragColor = base_color;
                    }
                }
        
    '''

    VERT_2D_INSTANCED = '''
        #version 330
        
        uniform mat4 projection;
        uniform mat4 model;  // Add transform support
        
        in vec2 in_position;  // Unit circle vertices
        
        // Per-instance attributes
        in vec2 instance_center;
        in float instance_radius;
        in vec4 instance_color;
        
        out vec4 v_color;
        
        void main() {
            // Transform unit circle to local space
            vec2 local_pos = in_position * instance_radius + instance_center;
            
            // Apply model transform, then projection
            gl_Position = projection * model * vec4(local_pos, 0.0, 1.0);
            v_color = instance_color;
        }
    '''


    VERT_3D_INSTANCED_LINE = '''
        #version 330
        
        uniform mat4 projection;
        uniform mat4 view;
        
        in float in_position;  // 0.0 or 1.0
        
        // Per-instance attributes
        in vec3 instance_start;
        in vec3 instance_end;
        in vec4 instance_color;
        
        out vec4 v_color;
        
        void main() {
            // Interpolate between start and end
            vec3 world_pos = mix(instance_start, instance_end, in_position);
            gl_Position = projection * view * vec4(world_pos, 1.0);
            v_color = instance_color;
        }
    '''

    FRAG_3D_INSTANCED_LINE = '''
        #version 330
        
        in vec4 v_color;
        out vec4 fragColor;
        
        void main() {
            fragColor = v_color;
        }
    '''

        # For thick 3D lines
    VERT_3D_INSTANCED_THICK_LINE = '''
        #version 330
        
        uniform mat4 projection;
        uniform mat4 view;
        
        in vec3 in_position;  // Unit tube geometry (x from 0-1, yz from -0.5 to 0.5)
        in vec3 in_normal;
        
        // Per-instance attributes
        in vec3 instance_start;
        in vec3 instance_end;
        in float instance_thickness;
        in vec4 instance_color;
        
        out vec4 v_color;
        out vec3 v_normal;
        
        void main() {
            // Calculate line direction
            vec3 direction = instance_end - instance_start;
            float line_length = length(direction);
            direction = normalize(direction);
            
            // Find perpendicular vectors for the tube cross-section
            vec3 up = abs(direction.y) > 0.99 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
            vec3 right = normalize(cross(direction, up));
            up = normalize(cross(right, direction));
            
            // Scale the cross-section by thickness
            right *= instance_thickness;
            up *= instance_thickness;
            
            // Transform the unit tube vertex to world space
            // in_position.x goes from 0 to 1 along the line
            // in_position.y and in_position.z are the cross-section offsets
            vec3 world_pos = instance_start 
                        + direction * (in_position.x * line_length)
                        + right * in_position.y
                        + up * in_position.z;
            
            gl_Position = projection * view * vec4(world_pos, 1.0);
            v_color = instance_color;
            
            // Transform normal (simplified - for proper lighting would need full transform)
            v_normal = in_normal.x * direction + in_normal.y * normalize(right) + in_normal.z * normalize(up);
        }
    '''

    FRAG_3D_INSTANCED_THICK_LINE = '''
        #version 330
        
        in vec4 v_color;
        in vec3 v_normal;
        out vec4 fragColor;
        
        void main() {
            fragColor = v_color;
        }
    '''

    FRAG_2D_INSTANCED = '''
        #version 330
        
        in vec4 v_color;
        out vec4 fragColor;
        
        void main() {
            fragColor = v_color;
        }
    '''

    VERT_2D_INSTANCED_LINE = '''
        #version 330
        
        uniform mat4 projection;
        uniform mat4 model;
        
        in float in_position;  // 0.0 or 1.0
        
        // Per-instance attributes
        in vec2 instance_start;
        in vec2 instance_end;
        in vec4 instance_color;
        
        out vec4 v_color;
        
        void main() {
            // Interpolate between start and end
            vec2 world_pos = mix(instance_start, instance_end, in_position);
            gl_Position = projection * model * vec4(world_pos, 0.0, 1.0);
            v_color = instance_color;
        }
    '''

    FRAG_2D_INSTANCED_LINE = '''
        #version 330
        
        in vec4 v_color;
        out vec4 fragColor;
        
        void main() {
            fragColor = v_color;
        }
    '''

    # For thick lines - use rectangle shader with different geometry
    VERT_2D_INSTANCED_THICK_LINE = '''
        #version 330
        
        uniform mat4 projection;
        uniform mat4 model;
        
        in vec2 in_position;  // Unit thick line geometry vertices
        
        // Per-instance attributes  
        in vec2 instance_start;
        in vec2 instance_end;
        in float instance_thickness;
        in vec4 instance_color;
        
        out vec4 v_color;
        
        void main() {
            // Calculate line direction and perpendicular
            vec2 direction = instance_end - instance_start;
            float length = length(direction);
            direction = direction / length;
            
            vec2 perpendicular = vec2(-direction.y, direction.x);
            
            // Transform unit line geometry
            // in_position.x is along the line (0 to 1)
            // in_position.y is perpendicular (-0.5 to 0.5)
            vec2 world_pos = instance_start 
                        + direction * (in_position.x * length)
                        + perpendicular * (in_position.y * instance_thickness);
            
            gl_Position = projection * model * vec4(world_pos, 0.0, 1.0);
            v_color = instance_color;
        }
    '''

    VERT_2D_INSTANCED_RECT = '''
        #version 330
        
        uniform mat4 projection;
        uniform mat4 model;
        
        in vec2 in_position;  // Unit rectangle (0,0 to 1,1)
        
        // Per-instance attributes
        in vec2 instance_pos;    // Top-left corner
        in vec2 instance_size;   // Width and height
        in vec4 instance_color;
        
        out vec4 v_color;
        
        void main() {
            // Transform unit rectangle to world space
            vec2 world_pos = in_position * instance_size + instance_pos;
            gl_Position = projection * model * vec4(world_pos, 0.0, 1.0);
            v_color = instance_color;
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
                in vec2 in_texcoord_0;
                
                out vec2 v_texcoord;
                
                void main() {
                    v_texcoord = in_texcoord_0;
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
                in vec2 in_texcoord_0;
                
                out vec2 v_texcoord;
                
                void main() {
                    v_texcoord = in_texcoord_0;
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
                    in vec2 in_texcoord_0;
                    
                    out vec2 v_texcoord;
                    
                    void main() {
                        v_texcoord = in_texcoord_0;
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