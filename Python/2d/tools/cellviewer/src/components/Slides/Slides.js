/*

Props:
- imgs      = [[imageSrc,framename]]    (main)
- frames    = {} from frames.JSON       (radialtree)
- colony    = {} from colony.JSON       (parseColony)
- angles    = {} from angles.JSON       (radialtree)

*/


import React, { Component } from 'react';
import { Row } from 'reactstrap';
import ImageCell from './Image/Image';
import Slider from '@material-ui/lab/Slider';
import PlayIcon from '@material-ui/icons/PlayArrow';
import PauseIcon from '@material-ui/icons/Pause';
import Fab from '@material-ui/core/Fab';
import Grid from '@material-ui/core/Grid';

const status_map = {
    "play": 1,
    "pause": 0
}

export default class Slides extends Component {
    constructor(props) {
        super(props);
        this.state = {
            tog: 0
        };

        this.pos = 0;

        this.status = "play";

        this.images = props.imgs;
        this.colony = props.colony;
        this.change = this.change.bind(this);
        this.play = this.play.bind(this);
        this.click = this.click.bind(this);
        this.spaceBar = this.spaceBar.bind(this);
        window.onkeyup = this.spaceBar;
    }

    spaceBar(e){
        var key = e.keyCode ? e.keyCode : e.which;
        if (key === 32) {
            if (e.stopPropagation) {
                e.stopPropagation();
                e.preventDefault();
            }
            this.click();
        }
    }

    change(e,i) {
        if(i>=0 && i<this.images.length){
            this.pos = i;
            this.setState({
                tog : i
            });
        }
    }

    play(){
        if(this.status==="pause"){
            setTimeout(function() {
                this.change(null,(this.pos+1)%this.images.length);
                this.play();
            }.bind(this), 300);
        }
    }

    click(){
        if(this.status ==="pause"){
            this.status = "play"
            this.setState({
                tog : 0
            })
        }else{
            this.status = "pause"
            this.play()
        }
    }

    render() {
        if(this.images.length!==this.props.imgs.length){
            this.pos = 0;
        }
        this.images = this.props.imgs;
        this.colony = this.props.colony;
        return (
            <div>
                <Row>
                    <ImageCell
                        curr={this.pos}
                        src={this.images[this.pos]} 
                        colony={this.colony[this.pos.toString()]["cells"]}
                        pos={(this.pos+1.5)/(this.images.length+2)}
                        colors = {this.props.colors}
                        angles={this.props.angles}
                        frames={this.props.frames}/>

                    {/* --------------------------  Player and Progress Bar-------------------------- */}
                    <Grid container direction="row" justify="center" alignItems="center" style={{
                            position:"absolute",
                            bottom: "4.5%"
                            }}>
                        <Grid item xs={1}>
                            <Fab onClick={this.click}
                                size="medium" aria-label="Add">
                                {[<PauseIcon />,<PlayIcon />][status_map[this.status]]}
                            </Fab>
                        </Grid>
                        
                        <Grid item xs={10}>
                            <Slider 
                                    value={this.pos}
                                    style={{width:"100%"}}
                                    aria-labelledby="discrete-slider"
                                    step={1}
                                    onChange={this.change}
                                    min={0}
                                    max={this.images.length-1}/>
                        </Grid>
                    </Grid>
                </Row>
            </div>
        );
    }
}
