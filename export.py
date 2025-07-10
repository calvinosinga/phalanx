import json

def export(scene, out_dir):
    # TODO if out_dir doesn't exist, create it

    graphics = scene.getGraphics()
    # TODO check that we don't have name repeats

    style = {}
    for gph in graphics:
        # we write vtp files for each graphic individually
        gph.writeVTP(out_dir)
        
        # we collect style jsons and output a single file after the loop
        style[gph.getName()] = gph.getStyle()
        
    #TODO save style to json file
    
    return