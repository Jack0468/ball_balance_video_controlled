class TargetStateMachine:
    def __init__(self):
        self.current_target_name = "center"
        self.valid_targets = ["center", "blue", "green", "red", "yellow"]
        
    def process_command(self, command):
        if command is None:
            return
            
        if command == "hold" or command == "stop":
            print(f"[{command.upper()}] Defaulting to center!")
            self.current_target_name = "center"
        elif command.startswith("go_"):
            color = command.split("_")[1]
            if color in self.valid_targets:
                print(f"[GO {color.upper()}] Switching target to {color} marker!")
                self.current_target_name = color
                
    def get_target_coords(self, marker_coords):
        if self.current_target_name == "center":
            return 0.0, 0.0
            
        # Target is a color. Do we see it?
        if self.current_target_name in marker_coords:
            return marker_coords[self.current_target_name]
            
        # If we can't see the target we were told to go to, 
        # default to center as a fallback to prevent flying off.
        return 0.0, 0.0
