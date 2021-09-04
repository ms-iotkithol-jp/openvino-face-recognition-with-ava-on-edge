from azure.iot.device import IoTHubModuleClient, IoTHubDeviceClient

class IoTClient:
    def __init__(self):
        pass

    def connect(self):
        pass

    def patch_twin_reported_properties(self, patch):
        pass

    def receive_twin_desired_properties_patch_handler(self, handler, lock):
        return None
    
    def get_twin(self):
        return None
    
    def send_message(self, msg, output_path):
        pass

class IoTModuleClient(IoTClient):
    def __init__(self):
        self.moduleClient = IoTHubModuleClient.create_from_edge_environment()
    
    def connect(self):
        self.moduleClient.connect()
    
    def patch_twin_reported_properties(self, patch):
        self.moduleClient.patch_twin_reported_properties(patch)

    def receive_twin_desired_properties_patch_handler(self, handler, lock):
        self.moduleClient.on_twin_desired_properties_patch_received = lambda p: handler(self.moduleClient, p, None, None, lock)
    
    def get_twin(self):
        return self.moduleClient.get_twin()
    
    def send_message(self, msg, output_path):
        if not output_path:
            output_path = 'output'
        self.moduleClient.send_message_to_output(msg, output_path)

class IoTDeviceClient(IoTClient):
    def __init__(self, connection_string):
        self.deviceClient = IoTHubDeviceClient.create_from_connection_string(connection_string)
    
    def connect(self):
        self.deviceClient.connect()
    
    def patch_twin_reported_properties(self, patch):
        self.deviceClient.patch_twin_reported_properties(patch)

    def receive_twin_desired_properties_patch_handler(self, handler, lock):
        self.deviceClient.on_twin_desired_properties_patch_received = lambda p: handler(self.deviceClient, p, None, None, lock)
    
    def get_twin(self):
        return self.deviceClient.get_twin()
    
    def send_message(self, msg, output_path):
        self.deviceClient.send_message(msg)
