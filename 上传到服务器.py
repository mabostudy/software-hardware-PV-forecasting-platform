import paramiko,datetime,os,time

hostname= '49.232.44.195'
username='ubuntu'
password=''  ##密码省略
port=22
remote_dir='/home/ubuntu'


try:
    t=paramiko.Transport((hostname,port))
    t.connect(username=username,password=password)
    sftp=paramiko.SFTPClient.from_transport(t)
    folder_name = time.strftime("%Y-%m-%d", time.localtime())
    local_dir=os.path.join('/home/mabo/Desktop/PvImages',folder_name)
 
    remote_dir = '/home/ubuntu/{}/'.format(folder_name)

    sftp.mkdir(remote_dir) 
    files=os.listdir(local_dir)
    for f in files:
        sftp.put(os.path.join(local_dir,f),os.path.join(remote_dir,f))
    t.close()
except Exception as e:
    pass
