import io
import os
import base64

import matplotlib.pyplot as plt
from matplotlib import rcParams

import settings

rcParams.update({'figure.autolayout': True})

_DPI = 96
_figure, _axix = plt.subplots()


def get_loss_plot(train_loss, valid_loss=[]):
    _axix.plot(train_loss, label='Training Loss')
    _axix.plot(valid_loss, label='Validation Loss')
    _axix.set_title('Loss vs Epochs')
    _axix.set_xlabel('Epochs')

    _figure.set_size_inches(500/_DPI, 500/_DPI)

    io_bytes = io.BytesIO()
    plt.savefig(io_bytes, format='png')
    plt.cla()
    io_bytes.seek(0)
    base64_image = base64.b64encode(io_bytes.getvalue()).decode('utf-8')
    io_bytes.close()
    return base64_image

def get_predict_plot(xy_truth, xy_test, filtered=False):
    '''
        given xy_truth = [x_truth,y_truth] and xy_test = [x_test,y_test]
        two lists containing the list of prediction coordenates, 
        returns image64 of the plot 
    '''

    x_truth = xy_truth[0]
    y_truth = xy_truth[1]
    x_test = xy_test[0]
    y_test = xy_test[1]

    if(filtered):
        lbl = "Ground Truth Filtered"
    else:
        lbl = "Ground Truth (raw)"
    _axix.plot(x_truth,y_truth,  label=lbl )
    _axix.plot(x_test, y_test, 'g--', label='Test Predictions')
    _axix.legend()

    _figure.set_size_inches(1000/_DPI, 450/_DPI)

    io_bytes = io.BytesIO()
    plt.savefig(io_bytes, format='png')
    plt.cla()
    io_bytes.seek(0)
    base64_image = base64.b64encode(io_bytes.getvalue()).decode('utf-8')
    io_bytes.close()
    return base64_image


def generate_machine_html(list_loss_plot, list_pred_plot, list_config):
    '''
    

    '''
    
    html = "<div>"
    for loss_plot, pred_plot, config in zip(list_loss_plot,list_pred_plot,list_config):
        [train_loss,valid_loss] = loss_plot
        [xy_truth_input, xy_test, y_trut_filt] = pred_plot

        fig64_loss_plot = get_loss_plot(train_loss, valid_loss)

        fig64_pred_ground_truth_as_input = get_predict_plot(xy_truth_input, xy_test)

        fig64_pred_ground_truth_filtered = get_predict_plot([xy_truth_input[0],y_trut_filt], xy_test, filtered=True)

        figs_one_set_config = [fig64_loss_plot,
                            fig64_pred_ground_truth_as_input,
                            fig64_pred_ground_truth_filtered]

        html += '<div class="row" style="white-space:nowrap; vertical-align: top;font-family: calibri">'
        html += (
            '<div class="column" style="display: inline-block;vertical-align: top;">' +
                config +
            '</div>' 
        )

        for figure_b64 in figs_one_set_config:
            html += (
                '<div class="column" style="display: inline-block;vertical-align: top;">' +
                    '<img src="data:image/png;base64,{0}">'.format(figure_b64) +
                '</div>'
            )
        html += '</div>'

    html += '</div>'

    html_path = '{0}/predictions_{1}.html'.format(settings.DIRECTORY,settings.NAME)

    # Create output directory if not existing
    if not os.path.isdir(settings.DIRECTORY):
        os.makedirs(settings.DIRECTORY)

    with open(html_path, 'w', errors='replace') as html_file:
        html_file.write(html)