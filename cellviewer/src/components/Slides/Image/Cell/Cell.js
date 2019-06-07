import React, { Component } from 'react';
import { Tooltip } from 'reactstrap';

export default class Cell extends Component {
    constructor(props) {
        super(props);
        this.toggle = this.toggle.bind(this);
        this.state = {
            tooltipOpen: false
        };
    }

    toggle() {
        this.setState({
            tooltipOpen: !this.state.tooltipOpen
          });
    }

    render() {
        return (
            <div>
                <div 
                    id={"Tooltip"+this.props.k}
                    style={{
                        position:"absolute",
                        top:this.props.top,
                        left:this.props.left,
                        backgroundColor:this.props.color,
                        width:"7px",
                        height:"7px"
                    }}/>
                <div 
                    id={"TooltipR"+this.props.k}
                    style={{
                        position:"absolute",
                        bottom:this.props.bottomR,
                        left:this.props.leftR,
                        backgroundColor:this.props.color,
                        width:"7px",
                        height:"7px"
                    }}/>
                <Tooltip placement="right" isOpen={this.state.tooltipOpen} target={"Tooltip"+this.props.k} toggle={this.toggle}>
                    <div 
                        style={{
                            marginLeft:"2px",
                            padding: "1px",
                            textAlign:"center",
                            backgroundColor:"yellow",
                            borderTopLeftRadius:"0px",
                            borderBottomLeftRadius:"0px",
                            borderTopRightRadius:"5px",
                            borderBottomRightRadius:"5px"}}>
                        {this.props.k}
                    </div>
                </Tooltip>
                <Tooltip placement="right" isOpen={this.state.tooltipOpen} target={"TooltipR"+this.props.k} toggle={this.toggle}>
                    <div 
                        style={{
                            marginLeft:"2px",
                            padding: "1px",
                            textAlign:"center",
                            backgroundColor:"yellow",
                            borderTopLeftRadius:"0px",
                            borderBottomLeftRadius:"0px",
                            borderTopRightRadius:"5px",
                            borderBottomRightRadius:"5px"}}>
                        {this.props.k}
                    </div>
                </Tooltip>
            </div>
        );
    }
}
