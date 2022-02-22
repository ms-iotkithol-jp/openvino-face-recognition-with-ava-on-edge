import os, uuid
import logging
import datetime
import asyncio
from azure.storage.blob import BlobServiceClient
import subprocess
from subprocess import PIPE


class FileUploader:
    def __init__(self, blobonedge_module_name, blob_account_name, blob_account_key, container_name):
        netinfo = self.getipaddress()
        hostname = blobonedge_module_name
        if 'wwan0' in netinfo:
            hostname = netinfo['wwan0']
            logging.info(f'hostname chaneged -> {hostname}')            
        elif 'wlan0' in netinfo:
            hostname = netinfo['wlan0']
            logging.info(f'hostname chaneged -> {hostname}')
        elif 'eth0' in netinfo:
            hostname = netinfo['eth0']
            logging.info(f'hostname chaneged -> {hostname}')
        connectionString = f'DefaultEndpointsProtocol=http;BlobEndpoint=http://{hostname}:11002/{blob_account_name};AccountName={blob_account_name};AccountKey={blob_account_key};'
        self.blobServiceClient = BlobServiceClient.from_connection_string(conn_str=connectionString, api_version='2017-04-17')
        logging.info(f'connected by connection-string={connectionString}')
        # self.blobServiceClient._X_MS_VERSION = '2017-04-17'
        self.containerClient = self.blobServiceClient.get_container_client(container_name)
        self.containerName = container_name
        self.hostname = hostname
    
    def initialize(self):
        try:
            self.containerClient.create_container()
            logging.info(f'created container={self.containerName}')
        except Exception as e:
            logging.error(f'failed to create container={self.containerName} - {e}')
    
    def upload(self, data, device_id, timestamp, ext_name):
        if self.containerClient:
            blobName = f'{device_id}/img-{timestamp}.{ext_name}'
            blobClient = self.containerClient.get_blob_client(blobName)
            try:
                blobClient.upload_blob(data)
                logging.info(f'uploaded blob - {blobName}')
            except Exception as e:
                logging.error(f'faild to upload blob={blobName} - {e}')

    def getipaddress(self):
        logging.info( 'Checking host IP address...')
        netinfo = {}
        for netname in ["eth0", "wlan0", "wwan0"]:
            proc = subprocess.run("/sbin/ip address show {}".format(netname), shell=True, stdout=PIPE)
            l = str(proc.stdout)
            logging.info(str(l))
            if l.find('inet') > 0:
                ipaddress = l[l.find('inet'):].split()[1]
                if ipaddress.find('/') > 0:
                    ipaddress = ipaddress.split('/')[0]
                netinfo[netname] = ipaddress
                logging.info('Found - {}'.format(ipaddress))
        return netinfo

    def gethostname(self):
        return self.hostname
