from distributed_trainer.data_dispatcher import DistributedIndexSamplerServer


if __name__ == "__main__":
    from rpyc.utils.server import ThreadedServer
    t = ThreadedServer(DistributedIndexSamplerServer, port=5003)
    print("Starting DistributedIndexSampler At Port 5003")
    t.start()

        
        
        