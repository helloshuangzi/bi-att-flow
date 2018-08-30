from hyperdrive import HyperDriveClient

hyperdriveClient = None

def initialize(args):
    global hyperdriveClient

    hyperdriveClient = HyperDriveClient(args=args)

def report(iteration_number, dev_results_official, dev_results, train_results, checkpoint_path):
    global hyperdriveClient

    metric = [
        {
            "sequence_number": int(iteration_number),
            "value": {
                "dev_exact_match_official": float(dev_results_official["exact_match"]),
                "dev_f1_official": float(dev_results_official["f1"]),
                "dev_accuracy": float(dev_results.acc),
                "dev_f1": float(dev_results.f1),
                "dev_loss": float(dev_results.loss),
                "train_accuracy": float(train_results.acc),
                "train_f1": float(train_results.f1),
                "train_loss": float(train_results.loss),
                "hyperdrive_model_checkpoint": str(checkpoint_path)
            }
        }
    ]    
    hyperdriveClient.report_metric(metric=metric)

def get_checkpoint_path():
    global hyperdriveClient

    return hyperdriveClient.get_model_checkpoint_path() if hyperdriveClient else None