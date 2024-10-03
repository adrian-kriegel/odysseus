'''
Utility for injecting metadata into generated code.
'''

def create_metadata_getter(func_name, data : dict[str, list[str]]):
    '''
    Creates a void function with an outparam std::unordered_map which populates the map with the keys and values from the metadata dict.

    Example: 
        create_metadata_getter('get_controller_metadata', {
            'joints': ['joint1', 'joint2'],
        })

    Results in a function with the following signature:
    
        void get_controller_metadata(std::unordered_map<std::vector<std::string>>& data);
    '''
    def to_cpp_initializer_list(l):

        return ', '.join([f'"{x}"' for x in l])

    assignments = ';\n\n  '.join(f'data["{name}"] = {{{to_cpp_initializer_list(value)}}}' for name, value in data.items())

    return f'''
void {func_name}(
    std::unordered_map<std::string, std::vector<std::string>>& data
){{
    {assignments};
}}
    '''

def create_metadata_file(getters : list[str]):
    '''
    Create a source file from getters generated via create_metadata_getter.
    '''

    getters_joined = '\n\n'.join(getters)

    return f'''
#include <string>
#include <unordered_map>
#include <vector>

extern "C" {{

{getters_joined}

}}
    '''