if __name__ == '__main__':
    import torchreid
    from torchreid.data.datasets.image.newdataset import NewDataSet

    torchreid.data.register_image_dataset('newdataset', NewDataSet)

    datamanager = torchreid.data.ImageDataManager(
        root='reid-data',
        sources='newdataset',
        targets='newdataset',
        height=256,
        width=128,
        transforms=['random_flip', 'color_jitter']
    )

    model = torchreid.models.build_model(
        name='osnet_x1_0',
        # num_classes=datamanager._num_train_pids,
        num_classes=0,
        # loss='softmax',
        pretrained=True
    )
    model = model.cuda()

    weight_path = 'log\osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth'
    torchreid.utils.load_pretrained_weights(model, weight_path)

    optimizer = torchreid.optim.build_optimizer(model, optim='adam', lr=0.0003)

    # scheduler = torchreid.optim.build_lr_scheduler(
    #  optimizer, lr_scheduler='single_step', stepsize=20
    # )

    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        # scheduler=scheduler,
        label_smooth=True
    )

    engine.run(
        save_dir='log/newdata_visrank',
        test_only=True,
        # max_epoch=60,
        # eval_freq=10,
        # print_freq=10,
        visrank=True,
        visrank_topk=30,
        # rerank=True
    )
