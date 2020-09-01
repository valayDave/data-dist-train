from distributed_trainer.data_dispatcher import DistributedIndexSampler


if __name__ == "__main__":
    from rpyc.utils.server import ThreadedServer
    t = ThreadedServer(DistributedIndexSampler, port=5003)
    print("Starting DistributedIndexSampler At Port 5003")
    t.start()

        
        
        