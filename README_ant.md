## Requirements

```
acme/acme/requirements.txt
```

## Run the code

### From scratch

```
python examples/tf/control_suite/lp_mpo.py --env=open_ant
```

This should run a back-and-forth within a circle of radius 1.0 meters, and origin 0.0
You can enable video saving as well.

### From a checkpoint

```
python examples/tf/control_suite/lp_mpo.py --acme_id=20261023-141711 --env=open_ant
```

To resume the same checkpoint on a different robot model, pass its MuJoCo XML:

```
python examples/tf/control_suite/lp_mpo.py --acme_id=20261023-141711 --env=open_ant --xml=/path/to/model.xml
```

An empty `--xml` keeps the Open Ant default model. This writes new values and checkpoints into the existing folder.
